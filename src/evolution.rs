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

/// If set below 1.0, individual trait values are capped to this value.
pub static mut TRAIT_MAX : f32 = 1.0;
// If set above 0.0, individual trait values become 0.0 if they fall below.
pub static mut TRAIT_MIN : f32 = 0.0;

/// The exponentiality of the annealing function (see evolve method).
pub static mut ANNEALING_EXP : f64 = 2.0;
/// Enable or disable extinction mode (see evolve method).
pub static mut EXTINCTION_MODE : bool = true;

// The minimum crossover coefficient for traits (experts only).
pub static mut CROSS_MIN : f64 = 0.5;

////////////////////////////////////////////////////////////////////////////////////////////////////

use std::usize;
use std::f32;
use std::mem;
use std::clone::Clone;
use std::cmp::Ordering;
use std::marker::Sync;

extern crate rand;
use self::rand::Rng;
use self::rand::SeedableRng;
use self::rand::seq::SliceRandom;

extern crate rand_xorshift;
use self::rand_xorshift::XorShiftRng;

extern crate rayon;
extern crate indicatif;
use self::indicatif::{ MultiProgress, ProgressBar, ProgressStyle, ProgressDrawTarget };

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// Objectives are instances capable of assessing the quality of optimization solutions
/// (a.k.a the "fitness" of "individuals" in the nomenclature of this module).
///
pub trait Objective
{
    ///
    /// The objective function (a.k.a. utility or fitness function) for the optimization.
    /// 
    /// It receives the "traits" (n-dimensional unit vector) of a solution candidate 
    /// and must return a number which is higher for solutions that are better.
    /// 
    /// The length of trait vectors is determined when creating a Species / Ecosystem (see below).
    /// 
    fn assess(&self, traits : &[f32]) -> f32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// An individual represents a specific optimization solution, i.e. vector of trait expressions. 
///
#[derive(Debug, Clone)]
pub struct Individual
{
    traits: Vec<f32>,
    fitness: f32
}

#[allow(dead_code)]
impl Individual
{
    ///
    /// Creates a new individual with a given number of traits in a random state.
    /// 
    fn new(num_traits : usize) -> Self
    {
        //Spawning own RNG so the constructor does not require one to be passed.
        //This method is only used during initial species creation, so this is acceptable.
        let mut rng = XorShiftRng::from_rng(rand::thread_rng()).unwrap();
        
        let mut traits = Vec::with_capacity(num_traits);
        for _ in 0..num_traits {
            traits.push(rng.gen_range(0.25, 1.0));
        }
        let mut individual = Individual { traits, fitness : f32::NAN };
        individual.normalize();
        return individual;
    }

    ///
    /// Performs a simple vector normalization to unit size. Does not truncate values.
    /// 
    fn normalize_scale(&mut self)
    {
        let sum : f32 = self.traits.iter().sum();
        let scale : f32 = 1.0 / sum;
        for value in self.traits.iter_mut() {
            *value *= scale;
        }
    }
    
    ///
    /// Ensures that the current trait expression sums up to 1.0,
    /// and that each entry respects the TRAIT_MIN / TRAIT_MAX constraints.
    /// 
    fn normalize(&mut self)
    {
        //begin with a naive normalization...
        self.normalize_scale();
        
        let trait_min;
        let trait_max;
        unsafe {
            trait_min = TRAIT_MIN;
            trait_max = TRAIT_MAX;
        }
        
        //perform iterative rebalance / normalize until both is satisfied...
        loop {
            let mut norm : f32 = 1.0;
            let mut sum : f32 = 0.0;
            let mut count : usize = self.traits.len();
            
            for value in self.traits.iter_mut() {
                if *value < trait_min {
                    *value = 0.0;
                } else if *value >= trait_max {
                    *value = trait_max;
                    norm -= trait_max;
                    count -= 1;
                } else {
                    sum += *value;
                }
            }
            
            if norm < 0.0 {
                //We had to spent more than the entire unit length on max trait values.
                //Let's rescale and try that again...
                self.normalize_scale();
                continue;
            }
            
            if (sum - norm).abs() < 1e-5 {
                return;
            }
            
            if sum == 0.0 {
                //all non-saturated values are zero -> just fill them with equal distribution
                let share = (norm / count as f32).min(trait_max);
                for value in self.traits.iter_mut() {
                    if *value == 0.0 {
                        *value = share;
                    }
                }
                return;
            }
            
            let scale : f32 = norm / sum;
            let mut complete = true;
            for value in self.traits.iter_mut() {
                if (*value > 0.0) && (*value < trait_max) {
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
    /// Creates a new individual by randomly mixing traits with three other individuals,
    /// according to differential evolution algorithm. The crossover coefficient [0..1]
    /// influences the probability and intensity of trait mixing. 
    /// 
    fn breed(&self, a: &Self, b: &Self, c: &Self, cross_coeff: f64, rng: &mut impl Rng) -> Self
    {
        let mut traits = self.traits.clone();
        
        let mut force_cross : usize;
        loop {
            //pick one non-zero trait that will be crossed definitely
            force_cross = rng.gen_range(0, traits.len());
            if self.traits[force_cross] != 0.0 {
                break;
            }
        }
        
        let cross_min;
        unsafe {
            cross_min = CROSS_MIN;
        }
        
        for i in 0..traits.len() {
            if (i == force_cross) || rng.gen_bool(cross_coeff) {
                traits[i] = a.traits[i] +
                    (b.traits[i] - c.traits[i]) * ((cross_min + cross_coeff) as f32);
            }
        }
        
        let mut individual = Individual { traits, fitness : f32::NAN };
        individual.normalize();
        return individual;
    }
    
    #[inline]
    pub fn get_fitness(&self) -> f32
    {
        return self.fitness;
    }
    
    #[inline]
    pub fn get_traits(&self) -> &Vec<f32>
    {
        return &self.traits;
    }
    
    ///
    /// Generates an associative vector of traits and their indices, sorted by decending value.
    /// 
    pub fn enumerate_traits_sorted(&self) -> Vec<(usize, f32)>
    {
        let mut res : Vec<_> = self.traits.iter().map(|x| *x).enumerate().collect();
        //Using stable sort as readability of results may be of interest.
        res.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        return res;
    }
}

impl Ord for Individual
{
    fn cmp(&self, other: &Self) -> Ordering {
        return self.fitness.partial_cmp(&other.fitness).unwrap_or(Ordering::Equal);
    }
}

impl PartialOrd for Individual
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        return self.fitness.partial_cmp(&other.fitness);
    }
}

impl PartialEq for Individual
{
    fn eq(&self, other: &Self) -> bool {
        return self.fitness == other.fitness;
    }
}

impl Eq for Individual {}

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// A species is a collection of individuals that evolve together by recombining traits.
///
#[derive(Debug, Clone)]
pub struct Species
{
    individuals: Vec<Individual>
}

#[allow(dead_code)]
impl Species
{
    ///
    /// Creates a new species of given size each with a given number of traits in a random state.
    /// 
    pub fn new(num_individuals : usize, num_traits : usize) -> Species
    {
        let mut individuals = Vec::with_capacity(num_individuals);
        for _ in 0..num_individuals {
            individuals.push(Individual::new(num_traits));
        }
        return Species { individuals };
    }
    
    ///
    /// Evolve this species for a given number of generations towards the given objective.
    /// Returns the best fittness achieved amongst all individuals.
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
    pub fn evolve(&mut self, gens : usize, objective: &impl Objective, prgb: &ProgressBar) -> f32
    {
        //Determine fitnesses of first generation...
        for individual in self.individuals.iter_mut() {
            individual.fitness = objective.assess(&individual.traits);
        }
        
        let start_size = self.individuals.len();
        
        let annealing_exp;
        let extinction_mode;
        unsafe {
            annealing_exp = ANNEALING_EXP;
            extinction_mode = EXTINCTION_MODE;
        }
        
        //We are using the XorShift RNG algo, which makes breeding about twice as fast.
        //However, note that breeding runtime is typically insignificant compared with assess().
        let mut rng = XorShiftRng::from_rng(rand::thread_rng()).unwrap();
        
        let mut best_fitness = f32::NEG_INFINITY;
        
        //Procreate, rinse, repeat...
        for gen in 0..gens {
            
            let heat = (1.0 - ((gen as f64) / (gens as f64))).powf(annealing_exp);
            
            let mut worst_fitness = f32::INFINITY;
            let mut worst_index = usize::max_value();
            
            let mut offspring = self.breed_all(heat, &mut rng);
            for individual in offspring.iter_mut() {
                individual.fitness = objective.assess(&individual.traits);
            }
            
            for i in 0..self.individuals.len() {
                let mut fitness = self.individuals[i].fitness;
                if offspring[i].fitness > fitness {
                    fitness = offspring[i].fitness;
                    //Memory swap avoids deep clone of individual trait vector.
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
            
            if extinction_mode {
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
    fn breed_all(&self, cross_coeff: f64, rng: &mut impl Rng) -> Vec<Individual>
    {
        let mut offspring = Vec::with_capacity(self.individuals.len());
        
        for individual in self.individuals.iter() {
            //Note: May choose itself as a mate - not ideal but not a big problem either,
            //since offspring traits are generated from the difference between mates.
            let mut mates = self.individuals.choose_multiple(rng, 3);
            offspring.push(
                individual.breed(
                    mates.next().unwrap(),
                    mates.next().unwrap(),
                    mates.next().unwrap(),
                    cross_coeff,
                    rng
                )
            );
        }
        
        return offspring;
    }
    
    ///
    /// Returns the individual with the highest fitness of this species.
    ///
    pub fn pick_fittest_individual(&self) -> &Individual
    {
        return self.individuals.iter().max().unwrap();
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
pub struct Ecosystem
{
    species: Vec<Species>
}

#[allow(dead_code)]
impl Ecosystem
{
    ///
    /// Creates a new ecosystem with a given number of species, individuals and traits.
    /// 
    pub fn new(num_species : usize, num_individuals : usize, num_traits : usize) -> Ecosystem
    {
        let mut species = Vec::with_capacity(num_species);
        for _ in 0..num_species {
            species.push(Species::new(num_individuals, num_traits));
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
    pub fn evolve(&mut self, gens : usize, objective: &(impl Objective + Sync))
    {
        let progress_style = ProgressStyle::default_bar()
            .template("{bar:80} {pos:>5}/{len:5} {msg}")
            .progress_chars("##-");
        
        rayon::scope(move |threads| {
            
            let multi_progress = MultiProgress::new();
            multi_progress.set_draw_target(ProgressDrawTarget::stdout());
            multi_progress.set_move_cursor(true);
            
            for species in self.species.iter_mut() {
                
                let progress = multi_progress.add(ProgressBar::new(gens as u64));
                progress.set_style(progress_style.clone());
                progress.set_position(0);
                progress.set_draw_delta((gens as u64) / 20);
                
                threads.spawn(move |_| {
                    let best_fitness = species.evolve(gens, objective, &progress);
                    let best_fitness_str = format!("{:.6}", best_fitness);
                    progress.finish_with_message(&best_fitness_str);
                });
            }
            
            multi_progress.join().unwrap();
        });
    }

    ///
    /// Returns the individual with the highest fitness per species.
    ///
    fn pick_fittest_individuals(&self) -> Vec<&Individual>
    {
        let mut fittest_per_species = Vec::with_capacity(self.species.len());
        for species in self.species.iter() {
            fittest_per_species.push(species.pick_fittest_individual());
        }
        return fittest_per_species;
    }
    
    ///
    /// Returns the individual with the highest fitness across all species.
    ///
    pub fn pick_fittest_individual(&self) -> &Individual
    {
        return self.pick_fittest_individuals().iter().max().unwrap();
    }

    ///
    /// Returns a pseudo-individual with traits which are the arithmetic mean
    /// between the traits of the fittest individuals across the fittest species.
    /// The fitness of the pseudo-individual is also averaged (strictly incorrect, see below).
    ///
    /// Note that this is not a mathmatically correct approach for a strict optimization problem,
    /// since the average coordinates of multiple maxima are not guaranteed to be another maximum!
    /// However, this can be a useful approach for obtaining more stable results, if the goal is
    /// merely to identify which traits have a higher impact on result quality across many runs.
    ///
    pub fn average_fittest_individuals(&self, num_species: usize) -> Individual
    {
        let mut fittest_per_species = self.pick_fittest_individuals();
        fittest_per_species.sort_unstable();
        fittest_per_species.reverse();
        fittest_per_species.truncate(num_species);
        
        let num_traits = fittest_per_species[0].traits.len();
        let mut traits = Vec::with_capacity(num_traits);
        
        let norm = 1.0 / (fittest_per_species.len() as f32);
        
        for i in 0..num_traits {
            let mut avg : f32 = 0.0;
            for individual in fittest_per_species.iter() {
                avg += individual.traits[i] * norm;
            }
            traits.push(avg);
        }
        
        let mut fitness : f32 = 0.0;
        for individual in fittest_per_species.iter() {
            fitness += individual.fitness * norm;
        }
        
        return Individual { traits, fitness };
    }
}
