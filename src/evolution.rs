/*!

Copyright (c) 2019 Ralf Strobel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

////////////////////////////////////////////////////////////////////////////////////////////////////

This module implements a variant of the differential evolution algorithm, combined with aspects of
simulated annealing, to find approximate solutions for difficult linear optimization problems.
The design of the algorithm is particularly aimed at resource allocation, i.e. situations in which
the challenge is to find the best distribution of a limited resource across a fixed set of options.
In a mathematical sense, all solution candidates are represented by an n-dimensional unit vector.

As typical for evolutionary algorithms, the provided objective function must be continuous.
Beyond that the implementation is domain-agnostic. Domain abstraction is achieved via the
"Objective" trait, which must be implemented and instantiated by the user application.

Running an optimization entails instantiating an "Ecosystem" of appropriate size,
calling its evolve() method with the Objective instance, and retrieving the "fittest" solution.
Please refer to the documentation of the specific traits below for details.

The following global variables can be altered before(!) execution to modify behavior...
*/

/// The exponentiality of the annealing function (see evolve method).
static ANNEALING_EXP: f64 = 2.0;
/// Enable or disable extinction mode (see evolve method).
static EXTINCTION_MODE: bool = true;

// The minimum crossover coefficient for genes (experts only).
static CROSS_MIN: f64 = 0.5;

////////////////////////////////////////////////////////////////////////////////////////////////////

use std::clone::Clone;
use std::cmp::Ordering;
use std::f32;
use std::f64;
use std::marker::Sync;
use std::mem;
use std::usize;

extern crate rand;
use self::rand::seq::SliceRandom;
use self::rand::Rng;
use self::rand::SeedableRng;

extern crate rand_xorshift;
use self::rand_xorshift::XorShiftRng;

extern crate indicatif;
extern crate rayon;
use self::indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// Objectives are instances capable of assessing the quality of optimization solutions
/// (a.k.a the "fitness" of "individuals" in the nomenclature of this module).
///
pub trait Objective: Sync {
    ///
    /// The objective function (a.k.a. utility or fitness function) for the optimization.
    ///
    /// It receives the "genome" (n-dimensional unit vector) of a solution candidate
    /// and must return a number which is higher for solutions that are better.
    ///
    /// The length of genome vectors is determined when creating a Species / Ecosystem (see below).
    ///
    fn assess(&self, genome: &[f32]) -> f32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// A genepool is a meta-descriptor for a set of available genes.
/// It defines which genes are available, as well as their possible expression ranges.
///
#[derive(Debug, Clone)]
pub struct Genepool {
    gene_minima: Vec<f32>,
    gene_maxima: Vec<f32>,
}

#[allow(dead_code)]
impl Genepool {
    ///
    /// Creates a genepool containing a given number of genes,
    /// all of which simply have the maximum possible range of 0..1
    ///
    pub fn new_unconstrained(num_genes: usize) -> Self {
        return Self::from_global_constraints(num_genes, 0.0, 1.0);
    }

    ///
    /// Creates a genepool containing a given number of genes,
    /// all of which share the same range constraints.
    ///
    pub fn from_global_constraints(num_genes: usize, gene_min: f32, gene_max: f32) -> Self {
        let mut gene_minima = Vec::with_capacity(num_genes);
        let mut gene_maxima = Vec::with_capacity(num_genes);
        for _ in 0..num_genes {
            gene_minima.push(gene_min);
            gene_maxima.push(gene_max);
        }
        return Genepool {
            gene_minima,
            gene_maxima,
        };
    }

    ///
    /// Creates a genepool for individual gene maxima, but with no minima (all 0.0).
    ///
    pub fn from_individual_maxima(gene_maxima: &Vec<f32>) -> Self {
        let gene_minima = vec![0.0; gene_maxima.len()];
        return Genepool {
            gene_minima,
            gene_maxima: gene_maxima.clone(),
        };
    }

    ///
    /// Creates a genepool from individual gene maxima and minima.
    ///
    pub fn from_individual_constraints(gene_minima: &Vec<f32>, gene_maxima: &Vec<f32>) -> Self {
        assert_eq!(gene_minima.len(), gene_maxima.len());
        return Genepool {
            gene_minima: gene_minima.clone(),
            gene_maxima: gene_maxima.clone(),
        };
    }

    ///
    /// Returns the number of defined genes (not the number of effective genes - i.e. max > 0.0)
    ///
    #[inline]
    pub fn len(&self) -> usize {
        return self.gene_maxima.len();
    }

    ///
    /// Expands the constraints of this genepool to include the other.
    ///
    fn expand_to_include(&mut self, other: &Genepool) {
        assert_eq!(self.len(), other.len());

        for (i, value) in self.gene_maxima.iter_mut().enumerate() {
            *value = value.max(other.gene_maxima[i]);
        }
        for (i, value) in self.gene_minima.iter_mut().enumerate() {
            *value = value.min(other.gene_minima[i]);
        }
    }
}

///
/// An individual represents a specific optimization solution, i.e. vector of gene expressions.
///
#[derive(Debug, Clone)]
pub struct Individual {
    genome: Vec<f32>,
    fitness: f32,
}

#[allow(dead_code)]
impl Individual {
    ///
    /// Creates a new individual with a given number of genes in a random state.
    ///
    fn new(num_genes: usize) -> Self {
        //Spawning own RNG so the constructor does not require one to be passed.
        //This method is only used during initial species creation, so this is acceptable.
        let mut rng = XorShiftRng::from_rng(rand::thread_rng()).unwrap();

        let mut genome = Vec::with_capacity(num_genes);
        for _ in 0..num_genes {
            genome.push(rng.gen_range(0.25, 1.0));
        }
        return Individual {
            genome,
            fitness: f32::NAN,
        };
    }

    ///
    /// Performs a simple vector normalization to unit size. Does not truncate values.
    ///
    fn normalize_scale(&mut self) {
        let sum: f32 = self.genome.iter().sum();
        let scale: f32 = 1.0 / sum;
        for value in self.genome.iter_mut() {
            *value *= scale;
        }
    }

    ///
    /// Ensures that the current gene expressions sum up to 1.0,
    /// and that each entry respects the constraints of the given genepool.
    ///
    fn normalize(&mut self, genepool: &Genepool) {
        //begin with a naive normalization...
        self.normalize_scale();

        //perform iterative rebalance / normalize until both is satisfied...
        loop {
            let mut norm: f32 = 1.0;
            let mut sum: f32 = 0.0;
            let mut count: usize = self.genome.len();

            for (i, value) in self.genome.iter_mut().enumerate() {
                let gene_min = genepool.gene_minima[i];
                if *value < gene_min {
                    *value = 0.0;
                    continue;
                }

                let gene_max = genepool.gene_maxima[i];
                if *value >= gene_max {
                    *value = gene_max;
                    norm -= gene_max;
                    count -= 1;
                    continue;
                }

                sum += *value;
            }

            if (sum - norm).abs() < 1e-5 {
                return;
            }

            if norm < 0.0 {
                //We had to spent more than the entire unit length on max gene values.
                //Let's rescale and try that again...
                self.normalize_scale();
                continue;
            }

            if sum == 0.0 {
                //all non-saturated values are zero -> just fill them with equal distribution
                let share = norm / count as f32;
                for (i, value) in self.genome.iter_mut().enumerate() {
                    if *value == 0.0 {
                        let gene_max = genepool.gene_maxima[i];
                        *value = share.min(gene_max);
                    }
                }
                continue;
            }

            let scale: f32 = norm / sum;
            let mut complete = true;
            for (i, value) in self.genome.iter_mut().enumerate() {
                let gene_max = genepool.gene_maxima[i];
                if (*value > 0.0) && (*value < gene_max) {
                    *value *= scale;
                    complete = false;
                }
            }
            if complete {
                return;
            }
        }
    }

    ///
    /// Creates a new individual by randomly mixing genes with three other individuals,
    /// according to differential evolution algorithm. The crossover coefficient [0..1]
    /// influences the probability and intensity of gene mixing.
    ///
    fn breed(&self, a: &Self, b: &Self, c: &Self, cross_coeff: f64, rng: &mut impl Rng) -> Self {
        let mut genome = self.genome.clone();

        let mut force_cross: usize;
        loop {
            //pick one non-zero gene that will be crossed definitely
            force_cross = rng.gen_range(0, genome.len());
            if self.genome[force_cross] != 0.0 {
                break;
            }
        }

        for i in 0..genome.len() {
            if (i == force_cross) || rng.gen_bool(cross_coeff) {
                genome[i] =
                    a.genome[i] + (b.genome[i] - c.genome[i]) * ((CROSS_MIN + cross_coeff) as f32);
            }
        }

        return Individual {
            genome,
            fitness: f32::NAN,
        };
    }

    #[inline]
    pub fn get_fitness(&self) -> f32 {
        return self.fitness;
    }

    #[inline]
    pub fn get_genome(&self) -> &Vec<f32> {
        return &self.genome;
    }

    ///
    /// Generates an associative vector of genes and their indices, sorted by decending value.
    ///
    pub fn enumerate_genes_sorted(&self) -> Vec<(usize, f32)> {
        let mut res: Vec<_> = self.genome.iter().map(|x| *x).enumerate().collect();
        //Using stable sort as readability of results may be of interest.
        res.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        return res;
    }
}

impl Ord for Individual {
    fn cmp(&self, other: &Self) -> Ordering {
        return self
            .fitness
            .partial_cmp(&other.fitness)
            .unwrap_or(Ordering::Equal);
    }
}

impl PartialOrd for Individual {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return self.fitness.partial_cmp(&other.fitness);
    }
}

impl PartialEq for Individual {
    fn eq(&self, other: &Self) -> bool {
        return self.fitness == other.fitness;
    }
}

impl Eq for Individual {}

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// A species is a collection of individuals that evolve together by recombining genes.
/// The evolution of genes within a species is constrained by the minima and maxima of its genepool.
///
/// Each species holds a dedicated genepool instance, which provides more flexibility
/// when aggregating individuals via a pseudo-species or complex multi-phase evolutions.
/// It also helps simplify the rust code by avoiding the need for lifetimes and thread-syncing.
///
#[derive(Debug, Clone)]
pub struct Species {
    genepool: Genepool,
    individuals: Vec<Individual>,
}

#[allow(dead_code)]
impl Species {
    ///
    /// Creates a new species with a given genepool and no individuals at first.
    ///
    pub fn new(genepool: Genepool) -> Self {
        return Species {
            genepool,
            individuals: Vec::new(),
        };
    }

    ///
    /// Creates the given number of individuals with genomes in a random state.
    ///
    pub fn spawn_individuals(&mut self, num_individuals: usize) {
        self.individuals.reserve_exact(num_individuals);
        for _ in 0..num_individuals {
            self.individuals.push(Individual::new(self.genepool.len()));
        }
    }

    ///
    /// Consumes and adds an externally created individual.
    /// Genome size must match the species genepool.
    ///
    pub fn adopt_individual(&mut self, individual: Individual) {
        assert_eq!(self.genepool.len(), individual.genome.len());
        self.individuals.push(individual);
    }

    ///
    /// Consumes an entire species by adopting all of its individuals.
    /// Genepools must match in size and are united.
    ///
    pub fn merge(&mut self, mut other: Species) {
        self.genepool.expand_to_include(&other.genepool);
        self.individuals.append(&mut other.individuals);
    }

    ///
    /// Evolve this species for a given number of generations towards the given objective.
    /// Returns the best fitness achieved amongst all individuals.
    ///
    /// The algorithm uses an exponential annealing function to gradually reduce the mutation rate
    /// of individuals, allowing for higher diversity in the beginning and increased precision
    /// towards the end of the evolution, when individuals should have settled into local optima.
    ///
    /// In "extinction" mode, the algorithm will also gradually decrease the number of individuals
    /// based on the annealing function, always discarding those with the lowest fitness.
    /// This will significantly improve performance with relatively small impact on result quality,
    /// but it leaves the species unusable for further evolve passes (multi-objective evolution).
    /// Also note that at most one individual will be discarded per generation, so that this mode
    /// should only be used when the number of generations exceeds the number of individuals.
    ///
    pub fn evolve(
        &mut self,
        objective: &dyn Objective,
        num_generations: usize,
        prgb: &ProgressBar,
    ) -> f32 {
        //Determine fitnesses of first generation...
        for individual in self.individuals.iter_mut() {
            individual.normalize(&self.genepool);
            individual.fitness = objective.assess(&individual.genome);
        }

        let start_size = self.individuals.len();

        //We are using the XorShift RNG algo, which makes breeding about twice as fast.
        //However, note that breeding runtime is typically insignificant compared with assess().
        let mut rng = XorShiftRng::from_rng(rand::thread_rng()).unwrap();

        let mut best_fitness = f32::NEG_INFINITY;

        //Procreate, rinse, repeat...
        for gen in 0..num_generations {
            let heat = (1.0 - ((gen as f64) / (num_generations as f64))).powf(ANNEALING_EXP);

            let mut worst_fitness = f32::INFINITY;
            let mut worst_index = usize::max_value();

            let mut offspring = self.breed_all(heat, &mut rng);
            for individual in offspring.iter_mut() {
                individual.normalize(&self.genepool);
                individual.fitness = objective.assess(&individual.genome);
            }

            for i in 0..self.individuals.len() {
                let mut fitness = self.individuals[i].fitness;
                if offspring[i].fitness > fitness {
                    fitness = offspring[i].fitness;
                    //Memory swap avoids deep clone of individual genome vector.
                    mem::swap(&mut self.individuals[i], &mut offspring[i]);
                }
                if fitness > best_fitness {
                    best_fitness = fitness;
                }
                if fitness < worst_fitness {
                    worst_fitness = fitness;
                    worst_index = i;
                }
            }

            if EXTINCTION_MODE {
                //Note: We must preserve at least 3 individuals for breeding to work.
                let num_dead = ((start_size - 3) as f64) * (1.0 - heat);
                let num_alive = start_size - (num_dead.round() as usize);
                if num_alive < self.individuals.len() {
                    self.individuals.swap_remove(worst_index);
                }
            }

            prgb.inc(1);
        }

        return best_fitness;
    }

    ///
    /// Creates the next generation of this species by breeding each individual with random mates.
    ///
    fn breed_all(&self, cross_coeff: f64, rng: &mut impl Rng) -> Vec<Individual> {
        let mut offspring = Vec::with_capacity(self.individuals.len());

        for individual in self.individuals.iter() {
            //Note: May choose itself as a mate - not ideal but not a big problem either,
            //since offspring genes are generated from the difference between mates.
            let mut mates = self.individuals.choose_multiple(rng, 3);
            offspring.push(individual.breed(
                mates.next().unwrap(),
                mates.next().unwrap(),
                mates.next().unwrap(),
                cross_coeff,
                rng,
            ));
        }

        return offspring;
    }

    ///
    /// Returns the individual with the highest fitness of this species.
    ///
    pub fn pick_fittest_individual(&self) -> &Individual {
        return self.individuals.iter().max().unwrap();
    }

    ///
    /// Create a pseudo-individual with a trait vector that is the arithmetic mean
    /// between all trait vectors of the individuals within the species,
    /// and a fitness that is the arithmetic mean of all individual fitnesses.
    ///
    /// Note that this is not a mathmatically correct approach for a strict optimization problem,
    /// since the average coordinates of multiple maxima are not guaranteed to be another maximum!
    /// However, this can be a useful approach for obtaining more stable results, if the goal is
    /// merely to identify which genes have a higher impact on result quality.
    ///
    pub fn create_average_individual(&self) -> Individual {
        let num_genes = self.genepool.len();
        let mut genome = Vec::with_capacity(num_genes);

        let norm = 1.0 / (self.individuals.len() as f32);

        for i in 0..num_genes {
            let mut avg: f32 = 0.0;
            for individual in self.individuals.iter() {
                avg += individual.genome[i] * norm;
            }
            genome.push(avg);
        }

        let mut fitness: f32 = 0.0;
        for individual in self.individuals.iter() {
            fitness += individual.fitness * norm;
        }

        return Individual { genome, fitness };
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// An ecosystem is a collection of species that evolve in parallel but independent of each other.
///
/// Evolving multiple species for the same optimization problem can be necessary if the topology
/// of the objective function contains many minor maxima in which individuals can get stuck.
///
#[derive(Debug, Clone)]
pub struct Ecosystem {
    species: Vec<Species>,
}

#[allow(dead_code)]
impl Ecosystem {
    ///
    /// Creates a new ecosystem with a given number of species and individuals.
    /// The borrowed genepool template is cloned into each species.
    ///
    pub fn new(genepool: &Genepool, num_species: usize, individuals_per_species: usize) -> Self {
        let mut species = Vec::with_capacity(num_species);
        for _ in 0..num_species {
            let mut created_species = Species::new(genepool.clone());
            created_species.spawn_individuals(individuals_per_species);
            species.push(created_species);
        }
        return Ecosystem { species };
    }

    ///
    /// Evolve all species for the same number of generations towards the same objective.
    /// -> Species::evolve()
    ///
    /// By default, physical CPU cores -1 threads are used to evolve multiple species in parallel,
    /// leaving one core for OS, worker queue and console progress bar rendering.
    /// The number of threads can be influenced using the RAYON_NUM_THREADS env variable.
    ///
    pub fn evolve(&mut self, objective: &dyn Objective, num_generations: usize) {
        let progress_style = ProgressStyle::default_bar()
            .template("{bar:80} {pos:>5}/{len:5} {msg}")
            .progress_chars("##-");

        rayon::scope(move |threads| {
            let multi_progress = MultiProgress::new();
            multi_progress.set_draw_target(ProgressDrawTarget::stdout());
            multi_progress.set_move_cursor(true);

            for species in self.species.iter_mut() {
                let progress = multi_progress.add(ProgressBar::new(num_generations as u64));
                progress.set_style(progress_style.clone());
                progress.set_position(0);
                progress.set_draw_delta((num_generations as u64) / 20);

                threads.spawn(move |_| {
                    let best_fitness = species.evolve(objective, num_generations, &progress);
                    let best_fitness_str = format!("{:.6}", best_fitness);
                    progress.finish_with_message(&best_fitness_str);
                });
            }

            multi_progress.join().unwrap();
        });
    }

    ///
    /// Creates a species from the fittest individuals of the N fittest species.
    /// The resulting genepool is the union of all source species genepools.
    ///
    pub fn create_species_of_fittest_individuals(&self, n_fittest_species: usize) -> Species {
        let mut genepool = self.species.first().unwrap().genepool.clone();
        let mut individuals = Vec::with_capacity(self.species.len());

        for species in self.species.iter() {
            genepool.expand_to_include(&species.genepool);
            individuals.push(species.pick_fittest_individual().clone());
        }

        individuals.sort_unstable();
        individuals.reverse();
        individuals.truncate(n_fittest_species);

        return Species {
            genepool,
            individuals,
        };
    }
}
