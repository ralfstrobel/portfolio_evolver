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

*/

use std::env;
use std::f32;

extern crate chrono;
use chrono::prelude::*;

mod storage;
use storage::*;

mod evolution;
use evolution::*;

////////////////////////////////////////////////////////////////////////////////////////////////////

const PORTFOLIO_ENV: &str = "PORTFOLIO";

const DATE_FROM_ENV: &str = "DATE_FROM";
const DATE_TO_ENV: &str = "DATE_TO";
const DATE_FORMAT: &str = "%Y-%m-%d";

const OBJECTIVE_ENV: &str = "OPTIMIZATION_TARGET";
const OBJECTIVE_NAME_MAX_PERF: &str = "max_performance";
const OBJECTIVE_NAME_MIN_DAY_LOSS: &str = "min_day_loss";
const OBJECTIVE_NAME_MIN_LOSS: &str = "min_loss";
const OBJECTIVE_NAME_MIN_LOSS_SUM: &str = "min_loss_sum";
const OBJECTIVE_NAME_MIN_LOSS_LENGTH: &str = "min_loss_length";
const OBJECTIVE_NAME_MIN_LOSS_AREA: &str = "min_loss_area";
const OBJECTIVE_NAME_MAX_GAIN_SUM: &str = "max_gain_sum";
const OBJECTIVE_NAME_MAX_GAIN_LENGTH: &str = "max_gain_length";
const OBJECTIVE_NAME_MAX_UNI: &str = "max_uni";
const OBJECTIVE_NAME_MAX_UNI_PERF: &str = "max_uni_perf";

const OBJECTIVE_SUFFIX_ENV: &str = "OPTIMIZATION_NAME_SUFFIX";
const SAVE_RESULTS_ENV: &str = "SAVE_RESULTS";

const NUM_SPECIES_ENV: &str = "NUM_SPECIES";
const NUM_SPECIES_DEFAULT: &str = "7";
const NUM_INDIVIDUALS_PER_GENE_ENV: &str = "NUM_INDIVIDUALS_PER_GENE";
const NUM_INDIVIDUALS_PER_GENE_DEFAULT: &str = "16";
const GENE_MAX_ENV: &str = "GENE_MAX";
const GENE_MAX_DEFAULT: &str = "0.15";

////////////////////////////////////////////////////////////////////////////////////////////////////

use std::alloc::{GlobalAlloc, Layout, System};

struct SimdAwareAllocator;
unsafe impl GlobalAlloc for SimdAwareAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let layout = Layout::from_size_align_unchecked(layout.size(), 32);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let layout = Layout::from_size_align_unchecked(layout.size(), 32);
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: SimdAwareAllocator = SimdAwareAllocator;

////////////////////////////////////////////////////////////////////////////////////////////////////

fn main() {
    //Initialize environment variables ...

    let portfolio_name = env::var(PORTFOLIO_ENV).unwrap();

    let start_date = match env::var(DATE_FROM_ENV) {
        Ok(date_from) => NaiveDate::parse_from_str(&date_from, DATE_FORMAT).unwrap(),
        Err(_) => NaiveDate::from_ymd(2010, 1, 1),
    };

    let end_date = match env::var(DATE_TO_ENV) {
        Ok(date_from) => NaiveDate::parse_from_str(&date_from, DATE_FORMAT).unwrap(),
        Err(_) => Utc::today().naive_local(),
    };

    let objective_name = env::var(OBJECTIVE_ENV).unwrap_or(String::from(OBJECTIVE_NAME_MAX_PERF));
    let objective_name_suffix = env::var(OBJECTIVE_SUFFIX_ENV).unwrap_or(String::new());

    let save_results = match env::var(SAVE_RESULTS_ENV) {
        Ok(save) => save == "1",
        Err(_) => true,
    };

    let num_species = env::var(NUM_SPECIES_ENV)
        .unwrap_or(String::from(NUM_SPECIES_DEFAULT))
        .parse::<usize>()
        .unwrap();

    let individuals_per_gene = env::var(NUM_INDIVIDUALS_PER_GENE_ENV)
        .unwrap_or(String::from(NUM_INDIVIDUALS_PER_GENE_DEFAULT))
        .parse::<usize>()
        .unwrap();

    let gene_maximum = env::var(GENE_MAX_ENV)
        .unwrap_or(String::from(GENE_MAX_DEFAULT))
        .parse::<f32>()
        .unwrap();

    //Load data...

    let mut portfolio = Portfolio::load(&portfolio_name);
    let asset_ids = portfolio.get_keys();
    let asset_titles = portfolio.get_titles();

    let asset_maxima: Vec<f32> = portfolio
        .get_mapped_property("max_percent", gene_maximum * 100.0)
        .iter()
        .map(|x| x * 0.01)
        .collect();

    println!("Loaded portfolio {}.", portfolio_name);
    //debug_log(&format!("{:?}\n", asset_ids));

    //Run Simulations...

    let master_genepool = Genepool::from_individual_maxima(&asset_maxima);

    let end_time: i64 = end_date.and_hms(0, 0, 0).timestamp() * 1000;
    let mut epoch_start_times: Vec<i64> = Vec::new();
    loop {
        let epoch_start_date = start_date
            .with_year(start_date.year() + (epoch_start_times.len() as i32))
            .unwrap();
        if epoch_start_date >= end_date {
            break;
        }
        epoch_start_times.push(epoch_start_date.and_hms(0, 0, 0).timestamp() * 1000);
    }

    let epoch_chart_sets =
        AlignedChartDataSet::load_epochs(&asset_ids, &epoch_start_times, end_time);

    let mut epoch_results = Species::new(master_genepool.clone());
    for charts in epoch_chart_sets.into_iter() {
        epoch_results.adopt_individual(run_simulation(
            &objective_name,
            &master_genepool,
            charts,
            num_species,
            individuals_per_gene,
        ));
    }

    let result = epoch_results.create_average_individual();

    //Print results...

    let mut result_text = String::new();
    result_text += &format!(
        "{}{} {:.6} = {{\n",
        objective_name,
        objective_name_suffix,
        result.get_fitness()
    );
    for (i, value) in result.enumerate_genes_sorted() {
        result_text += &format!("  {:.1}% {}\n", value * 100.0, asset_titles[i]);
    }
    result_text += &format!("}}");
    println!("\n{}", result_text);

    //Save results to file...
    if !save_results {
        return;
    }

    let result_property_name = format!("percent_{}{}", objective_name, objective_name_suffix);
    let result_property = result.get_genome().iter().map(|x| x * 100.0).collect();
    portfolio.set_mapped_property(&result_property_name, &result_property);
    portfolio.save();

    let cur_date_string = Local::today().format(DATE_FORMAT).to_string();
    let cur_year_string = Local::today().format("%Y").to_string();
    let portfolio_copy_name = format!("{}/{}_{}", cur_year_string, portfolio_name, cur_date_string);
    portfolio.set_name(&portfolio_copy_name);
    portfolio.save();
    result_text.push_str("\n\n");
    portfolio.save_text_log(&result_text);
}

fn run_simulation(
    objective_name: &str,
    master_genepool: &Genepool,
    charts: AlignedChartDataSet,
    num_species: usize,
    individuals_per_gene: usize,
) -> Individual {
    let num_genes = charts.len_non_empty();
    let num_individuals = individuals_per_gene * num_genes;
    let num_generations = num_individuals * 2;
    let num_results_averaged = (num_species / 2).max(1);

    println!(
        "\nLoaded {} data sets with {} data points each, spanning from {} to {}.",
        num_genes,
        charts.chart_len(),
        NaiveDateTime::from_timestamp(charts.get_start_timestamp() / 1000, 0).date(),
        NaiveDateTime::from_timestamp(charts.get_end_timestamp() / 1000, 0).date(),
    );
    println!(
        "Evolving {} species of {} individuals with {} genes for {} generations, selecting for '{}'...",
        num_species,
        num_individuals,
        num_genes,
        num_generations,
        objective_name
    );

    let mut genepool_mask = vec![1.0 as f32; master_genepool.len()];
    for i in charts.empty_indices().iter() {
        //mask out the charts which contain no data, so they will never be used
        genepool_mask[*i] = 0.0;
    }
    let mut genepool = master_genepool.clone();
    genepool.constrain_maxima(&genepool_mask);

    //println!("{:#.10?}", genepool);

    let mut ecosystem = Ecosystem::new(&genepool, num_species, num_individuals);
    match objective_name {
        OBJECTIVE_NAME_MAX_PERF => {
            let objective = MaxPerformanceObjective { charts };
            ecosystem.evolve(&objective, num_generations);
        }
        OBJECTIVE_NAME_MIN_DAY_LOSS => {
            let objective = MinDayLossObjective { charts };
            ecosystem.evolve(&objective, num_generations);
        }
        OBJECTIVE_NAME_MIN_LOSS => {
            let objective = MinLossObjective { charts };
            ecosystem.evolve(&objective, num_generations);
        }
        OBJECTIVE_NAME_MIN_LOSS_SUM => {
            let objective = MinLossSumObjective {
                charts,
                threshold: 0.01,
                exp: 1.2,
            };
            ecosystem.evolve(&objective, num_generations);
        }
        OBJECTIVE_NAME_MIN_LOSS_LENGTH => {
            let objective = MinLossLengthObjective {
                charts,
                threshold: 0.01,
                exp: 1.2,
            };
            ecosystem.evolve(&objective, num_generations);
        }
        OBJECTIVE_NAME_MIN_LOSS_AREA => {
            let objective = MinLossAreaObjective {
                charts,
                threshold: 0.01,
                exp: 1.2,
            };
            ecosystem.evolve(&objective, num_generations);
        }
        OBJECTIVE_NAME_MAX_GAIN_SUM => {
            let objective = MaxGainSumObjective { charts, exp: 0.8 };
            ecosystem.evolve(&objective, num_generations);
        }
        OBJECTIVE_NAME_MAX_GAIN_LENGTH => {
            let objective = MaxGainLengthObjective {
                charts,
                loss_tolerance: 0.025,
                exp: 1.0,
            };
            ecosystem.evolve(&objective, num_generations);
        }
        OBJECTIVE_NAME_MAX_UNI => {
            let objective = MaxUniformityObjective {
                charts,
                uni_exp: 1.0,
                perf_exp: 0.0,
            };
            ecosystem.evolve(&objective, num_generations);
        }
        OBJECTIVE_NAME_MAX_UNI_PERF => {
            let objective = MaxUniformityObjective {
                charts,
                uni_exp: 0.25,
                perf_exp: 4.0,
            };
            ecosystem.evolve(&objective, num_generations);
        }
        _ => panic!("Unknown objective: {}", objective_name),
    }
    //println!("{:#?}", ecosystem);

    let results = ecosystem.create_species_of_fittest_individuals(num_results_averaged);
    return results.create_average_individual();
}

struct MaxPerformanceObjective {
    charts: AlignedChartDataSet,
}
impl Objective for MaxPerformanceObjective {
    fn assess(&self, genome: &[f32]) -> f32 {
        let combined = self.charts.combine(&genome);
        let mut max = f32::NEG_INFINITY;
        for value in combined.iter() {
            max = value.max(max);
        }
        return max;
    }
}

struct MinDayLossObjective {
    charts: AlignedChartDataSet,
}
impl Objective for MinDayLossObjective {
    fn assess(&self, genome: &[f32]) -> f32 {
        let combined = self.charts.combine(&genome);

        let mut max_loss: f32 = 0.0;

        let mut prev = combined.first().unwrap();
        for value in combined.iter() {
            if value < prev {
                max_loss = max_loss.max(1.0 - (value / prev));
            }
            prev = value;
        }

        return -max_loss;
    }
}

struct MinLossObjective {
    charts: AlignedChartDataSet,
}
impl Objective for MinLossObjective {
    fn assess(&self, genome: &[f32]) -> f32 {
        let combined = self.charts.combine(&genome);

        let mut current_top = combined[0];
        let mut current_loss: f32 = 0.0;
        let mut result: f32 = 0.0;

        for value in combined.iter() {
            if *value < current_top {
                current_loss = current_loss.max(1.0 - (value / current_top));
            } else {
                current_top = *value;
                if current_loss != 0.0 {
                    result = result.max(current_loss);
                    current_loss = 0.0;
                }
            }
        }
        if current_loss != 0.0 {
            result = result.max(current_loss);
        }

        //debug_log(&format!("{:.10?} >> {:.5}\n", genome, deepest_dd));

        return -result;
    }
}

struct MinLossSumObjective {
    charts: AlignedChartDataSet,
    threshold: f64,
    exp: f64,
}
impl Objective for MinLossSumObjective {
    fn assess(&self, genome: &[f32]) -> f32 {
        let combined = self.charts.combine(&genome);

        let mut current_top = combined[0];
        let mut current_loss: f64 = 0.0;
        let mut result: f64 = 0.0;

        for value in combined.iter() {
            if *value < current_top {
                current_loss = current_loss.max(1.0 - ((*value as f64) / (current_top as f64)));
            } else {
                if current_loss >= self.threshold {
                    result += current_loss.powf(self.exp);
                }
                current_top = *value;
                current_loss = 0.0;
            }
        }
        if current_loss >= self.threshold {
            result += current_loss.powf(self.exp);
        }

        return -result as f32;
    }
}

struct MinLossLengthObjective {
    charts: AlignedChartDataSet,
    threshold: f64,
    exp: f64,
}
impl Objective for MinLossLengthObjective {
    fn assess(&self, genome: &[f32]) -> f32 {
        let combined = self.charts.combine(&genome);

        let mut current_top: f32 = combined[0];
        let mut current_top_time: usize = 0;
        let mut current_loss: f64 = 0.0;
        let mut current_loss_length: usize = 0;
        let mut result: f64 = 0.0;

        for (time, value) in combined.iter().enumerate() {
            if *value < current_top {
                current_loss = current_loss.max(1.0 - ((*value as f64) / (current_top as f64)));
                current_loss_length = time - current_top_time;
            } else {
                if current_loss >= self.threshold {
                    result += (current_loss_length as f64).powf(self.exp);
                }
                current_top = *value;
                current_top_time = time;
                current_loss = 0.0;
            }
        }
        if current_loss >= self.threshold {
            result += (current_loss_length as f64).powf(self.exp);
        }

        return -result as f32;
    }
}

struct MinLossAreaObjective {
    charts: AlignedChartDataSet,
    threshold: f64,
    exp: f64,
}
impl Objective for MinLossAreaObjective {
    fn assess(&self, genome: &[f32]) -> f32 {
        let combined = self.charts.combine(&genome);

        let mut current_top: f32 = combined[0];
        let mut result: f64 = 0.0;

        for value in combined.iter() {
            if *value < current_top {
                let current_loss = 1.0 - ((*value as f64) / (current_top as f64));
                if current_loss >= self.threshold {
                    result += current_loss.powf(self.exp);
                }
            } else {
                current_top = *value;
            }
        }

        return -result as f32;
    }
}

struct MaxGainSumObjective {
    charts: AlignedChartDataSet,
    exp: f64,
}
impl Objective for MaxGainSumObjective {
    fn assess(&self, genome: &[f32]) -> f32 {
        let combined = self.charts.combine(&genome);

        let mut current_top = combined[0];
        let mut result: f64 = 0.0;

        for value in combined.iter() {
            if *value > current_top {
                result += (((*value as f64) / (current_top as f64)) - 1.0).powf(self.exp);
                current_top = *value;
            }
        }

        return result as f32;
    }
}

struct MaxGainLengthObjective {
    charts: AlignedChartDataSet,
    loss_tolerance: f32,
    exp: f64,
}
impl Objective for MaxGainLengthObjective {
    fn assess(&self, genome: &[f32]) -> f32 {
        let combined = self.charts.combine(&genome);

        let mut current_top = combined[0];
        let mut current_gain_length: usize = 0;
        let mut result: f64 = 0.0;

        for value in combined.iter() {
            if *value > current_top {
                current_top = *value;
                current_gain_length += 1;
            } else {
                let current_loss = 1.0 - (*value / current_top);
                if current_loss > self.loss_tolerance {
                    result += (current_gain_length as f64).powf(self.exp);
                    current_gain_length = 0;
                }
            }
        }
        result += (current_gain_length as f64).powf(self.exp);
        return result as f32;
    }
}

struct MaxUniformityObjective {
    charts: AlignedChartDataSet,
    uni_exp: f64,
    perf_exp: f64,
}
impl Objective for MaxUniformityObjective {
    fn assess(&self, genome: &[f32]) -> f32 {
        let combined = self.charts.combine(&genome);

        let start_value = *combined.first().unwrap() as f64;
        let end_value = *combined.last().unwrap() as f64;
        let total_gain = ((end_value / start_value) - 1.0).max(0.0);

        let num_steps = combined.len() as f64;
        let avg_step_gain = total_gain / num_steps;

        let mut sqr_diff_sum: f64 = 0.0;
        let mut expected_gain: f64 = 0.0;
        for value in combined.iter() {
            let gain = ((*value as f64) / start_value) - 1.0;
            expected_gain += avg_step_gain;
            sqr_diff_sum += (gain - expected_gain).powf(2.0);
        }

        return ((total_gain + 1.0).powf(self.perf_exp) / sqr_diff_sum.powf(self.uni_exp)) as f32;
    }
}
