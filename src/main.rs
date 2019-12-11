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

use std::f32;
use std::env;

extern crate chrono;
use chrono::prelude::*;

mod storage;
use storage::*;

mod evolution;
use evolution::*;

////////////////////////////////////////////////////////////////////////////////////////////////////

const PORTFOLIO_ENV : &str = "PORTFOLIO";
const PORTFOLIO_DEFAULT : &str = "low_risk";

const DATE_FROM_ENV : &str = "DATE_FROM";
const DATE_TO_ENV : &str = "DATE_TO";
const DATE_FORMAT : &str = "%Y-%m-%d";

const OBJECTIVE_ENV : &str = "OPTIMIZATION_TARGET";
const OBJECTIVE_NAME_MAX_PERF : &str = "max_performance";
const OBJECTIVE_NAME_MIN_LOSS : &str = "min_loss";
const OBJECTIVE_NAME_MIN_DAY_LOSS : &str = "min_day_loss";
const OBJECTIVE_NAME_MIN_LOSS_SUM : &str = "min_loss_sum";
const OBJECTIVE_NAME_MAX_UNI : &str = "max_uni";

const OBJECTIVE_SUFFIX_ENV : &str = "OPTIMIZATION_NAME_SUFFIX";
const SAVE_RESULTS_ENV : &str = "SAVE_RESULTS";

const GENE_POPULATION_ENV: &str = "GENE_POPULATION";
const GENE_POPULATION_DEFAULT: &str = "16";
const GENE_MAX_ENV: &str = "GENE_MAX";
const GENE_MAX_DEFAULT: &str = "0.15";

////////////////////////////////////////////////////////////////////////////////////////////////////

use std::alloc::{GlobalAlloc, System, Layout};

struct SimdAwareAllocator;
unsafe impl GlobalAlloc for SimdAwareAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8
    {
        let layout = Layout::from_size_align_unchecked(layout.size(), 32);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout)
    {
        let layout = Layout::from_size_align_unchecked(layout.size(), 32);
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR : SimdAwareAllocator = SimdAwareAllocator;

////////////////////////////////////////////////////////////////////////////////////////////////////

fn main() {
    
    //Initialize environment variables ...
    
    let portfolio_name = env::var(PORTFOLIO_ENV)
        .unwrap_or(String::from(PORTFOLIO_DEFAULT));
    
    let from_time : i64 = match env::var(DATE_FROM_ENV) {
        Ok(date_from) => {
            NaiveDate::parse_from_str(&date_from, DATE_FORMAT).unwrap()
                .and_hms(0,0,0).timestamp() * 1000
        },
        Err(_) => 0
    };
    
    let to_times : i64 = match env::var(DATE_TO_ENV) {
        Ok(date_from) =>
            NaiveDate::parse_from_str(&date_from, DATE_FORMAT).unwrap()
                .and_hms(0,0,0).timestamp() * 1000,
        Err(_) => i64::max_value()
    };
    
    let objective_name = env::var(OBJECTIVE_ENV)
        .unwrap_or(String::from(OBJECTIVE_NAME_MAX_PERF));
    let objective_name_suffix = env::var(OBJECTIVE_SUFFIX_ENV)
        .unwrap_or(String::new());
    
    let save_results = match env::var(SAVE_RESULTS_ENV) {
        Ok(save) => save == "1",
        Err(_) => true
    };
    
    let gene_population_factor = env::var(GENE_POPULATION_ENV)
        .unwrap_or(String::from(GENE_POPULATION_DEFAULT))
        .parse::<usize>().unwrap();
    
    unsafe {
        evolution::GENE_MAX = env::var(GENE_MAX_ENV)
            .unwrap_or(String::from(GENE_MAX_DEFAULT))
            .parse::<f32>().unwrap();
    }
    
    //Load data...
    
    let mut portfolio = Portfolio::load(&portfolio_name);
    let asset_ids = portfolio.get_keys();
    let asset_titles = portfolio.get_titles();
    println!("Loaded portfolio {}.", portfolio_name);
    
    let charts = AlignedChartDataSet::load(&asset_ids, from_time, to_times);
    println!(
        "Loaded {} data sets with {} data points each, spaning from {} to {}.",
        charts.len(),
        charts.chart_len(),
        NaiveDateTime::from_timestamp(charts.get_start_timestamp() / 1000, 0).date(),
        NaiveDateTime::from_timestamp(charts.get_end_timestamp() / 1000, 0).date(),
    );
    
    //Run Simulation...
    
    let num_genes = charts.len();
    let num_individuals = gene_population_factor * num_genes;
    let num_generations = 2 * num_individuals;
    let num_species = 35;
    let num_results_averaged = 15;
    
    println!(
        "\nEvolving {} species of {} individuals with {} genes for {} generations, selecting for '{}'...",
        num_species,
        num_individuals,
        num_genes,
        num_generations,
        objective_name
    );
    
    let mut ecosystem = Ecosystem::new(num_species, num_individuals, num_genes);
    match objective_name.as_str() {
        OBJECTIVE_NAME_MAX_PERF => {
            let objective = MaxPerformanceObjective { charts };
            ecosystem.evolve(num_generations, &objective);
        },
        OBJECTIVE_NAME_MIN_LOSS => {
            let objective = MinLossObjective { charts };
            ecosystem.evolve(num_generations, &objective);
        },
        OBJECTIVE_NAME_MIN_DAY_LOSS => {
            let objective = MinDayLossObjective { charts };
            ecosystem.evolve(num_generations, &objective);
        },
        OBJECTIVE_NAME_MIN_LOSS_SUM => {
            let objective = MinLossSumObjective { charts };
            ecosystem.evolve(num_generations, &objective);
        },
        OBJECTIVE_NAME_MAX_UNI => {
            let objective = MaxUniformityObjective { charts };
            ecosystem.evolve(num_generations, &objective);
        },
        _ => panic!("Unknown objective: {}", objective_name),
    }
    //println!("{:#?}", ecosystem);
    
    //Print results...
    
    let result = ecosystem.average_fittest_individuals(num_results_averaged);
    //println!("{:#?}", result);
    
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

struct MaxPerformanceObjective { charts: AlignedChartDataSet }
impl Objective for MaxPerformanceObjective
{
    fn assess(&self, genome: &[f32]) -> f32
    {
        let combined = self.charts.combine(&genome);
        let mut max = f32::NEG_INFINITY;
        for value in combined.iter() {
            max = value.max(max);
        }
        return max;
    }
}

struct MinLossObjective { charts: AlignedChartDataSet }
impl Objective for MinLossObjective
{
    fn assess(&self, genome: &[f32]) -> f32
    {
        let combined = self.charts.combine(&genome);
        
        let mut current_max = combined[0];
        let mut current_dd : f32 = 0.0;
        let mut deepest_dd : f32 = 0.0;
        
        for value in combined.iter() {
            if *value < current_max {
                current_dd = current_dd.max(1.0 - (value / current_max));
            } else {
                current_max = *value;
                if current_dd != 0.0 {
                    deepest_dd = deepest_dd.max(current_dd);
                    current_dd = 0.0;
                }
            }
        }
        if current_dd != 0.0 {
            deepest_dd = deepest_dd.max(current_dd);
        }
        
        return -deepest_dd;
    }
}

struct MinDayLossObjective { charts: AlignedChartDataSet }
impl Objective for MinDayLossObjective
{
    fn assess(&self, genome: &[f32]) -> f32
    {
        let combined = self.charts.combine(&genome);
        
        let mut max_loss : f32 = 0.0;
        
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

struct MinLossSumObjective { charts: AlignedChartDataSet }
impl Objective for MinLossSumObjective
{
    fn assess(&self, genome: &[f32]) -> f32
    {
        let combined = self.charts.combine(&genome);
        
        let mut loss_sum : f64 = 0.0;
        
        let mut prev = *combined.first().unwrap();
        for value in combined.iter() {
            if *value < prev {
                loss_sum += 1.0 - ((*value as f64) / (prev as f64));
            }
            prev = *value;
        }
        
        return -loss_sum as f32;
    }
}

struct MaxUniformityObjective { charts: AlignedChartDataSet }
impl Objective for MaxUniformityObjective
{
    fn assess(&self, genome: &[f32]) -> f32
    {
        let combined = self.charts.combine(&genome);
        
        let start_value = *combined.first().unwrap() as f64;
        let end_value = *combined.last().unwrap() as f64;
        let total_gain = ((end_value / start_value) - 1.0).max(0.0);
        
        let num_steps = combined.len() as f64;
        let avg_step_gain = total_gain / num_steps;
        
        let mut sqr_diff_sum : f64 = 0.0;
        let mut expected_gain : f64 = 0.0;
        for value in combined.iter() {
            let gain = ((*value as f64) / start_value) - 1.0;
            expected_gain += avg_step_gain;
            sqr_diff_sum += (gain - expected_gain).powf(2.0);
        }
        
        return -sqr_diff_sum as f32;
    }
}
