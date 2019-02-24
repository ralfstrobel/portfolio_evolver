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

This module translates json file contents (portfolios, chart data) to and from runtime models.

*/

use std::fs::File;
use std::fs::OpenOptions;
use std::io::Read;
use std::io::Write;

extern crate serde;
use self::serde::Serialize;

extern crate serde_json;
use self::serde_json::{
    Value as JsonValue,
    Map as JsonObject,
    Serializer as JsonSerializer,
    ser::PrettyFormatter as JsonPrettyFormatter
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

////////////////////////////////////////////////////////////////////////////////////////////////////

const PORTFOLIO_DIRECTORY : &str = "../portfolios/";
const DATA_DIRECTORY : &str = "../data/";

const EXTENSION_JSON : &str = ".json";
const EXTENSION_TEXT_LOG : &str = ".txt"; 

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// A portfolio is a meta property container which describes a collection of assets.
/// It takes the form of a two-dimensional map <asset_id> => <property_name> => <value>.
///
#[derive(Debug, Clone)]
pub struct Portfolio
{
    file_name: String,
    json: JsonObject<String, JsonValue>
}

#[allow(dead_code)]
impl Portfolio
{
    ///
    /// Loads a portfolio JSON file from the conventional location in the portfolios directory.
    ///
    pub fn load(name: &str) -> Self
    {
        let mut file_name = String::from(PORTFOLIO_DIRECTORY);
        file_name.push_str(&name);
        file_name.push_str(EXTENSION_JSON);
        
        let mut file = File::open(&file_name).unwrap();
        let mut content = String::new();
        file.read_to_string(&mut content).unwrap();
        let json : JsonValue = serde_json::from_str(&content).unwrap();
        let json = json.as_object().unwrap().clone();
        
        return Self { file_name, json };
    }
    
    ///
    /// Changes the name of the portfolio. Use in combination with save() to write to new file.
    ///
    pub fn set_name(&mut self, new_name: &str)
    {
        self.file_name = String::from(PORTFOLIO_DIRECTORY);
        self.file_name.push_str(&new_name);
        self.file_name.push_str(EXTENSION_JSON);
    }
    
    ///
    /// Re-serializes the portfolio into a JSON string.
    ///
    fn to_string(&self) -> String
    {
        let buf = Vec::new();
        let formatter = JsonPrettyFormatter::with_indent(b"    ");
        let mut ser = JsonSerializer::with_formatter(buf, formatter);
        self.json.serialize(&mut ser).unwrap();
        return String::from_utf8(ser.into_inner()).unwrap();
    }
    
    ///
    /// Saves a portfolio, overwriting the file it was loaded from.
    ///
    pub fn save(&self)
    {
        let content = self.to_string();
        let mut file = File::create(&self.file_name).unwrap();
        file.write_all(content.as_bytes()).unwrap();
    }
    
    ///
    /// Creates or appends a text log of the same name next to the portfolio data file.
    ///
    pub fn save_text_log(&self, text: &str)
    {
        let mut file_name = self.file_name.clone();
        let file_base_name_len = file_name.len() - EXTENSION_JSON.len();
        file_name.truncate(file_base_name_len);
        file_name.push_str(EXTENSION_TEXT_LOG);
        let mut file = OpenOptions::new().create(true).append(true).open(&file_name).unwrap();
        file.write_all(text.as_bytes()).unwrap();
    }
    
    ///
    /// Returns the keys (asset IDs) of all assets referenced by this portfolio.
    ///
    pub fn get_keys(&self) -> Vec<String>
    {
        let mut keys = Vec::new();
        for key in self.json.keys() {
            let item = self.json.get(key).unwrap().as_object().unwrap();
            if !item.contains_key("percent") {
                continue;
            }
            keys.push(key.clone());
        }
        return keys;
    }
    
    ///
    /// Collects the "title" property across all assets within the portfolio.
    /// The returned vector aligns with the keys returned by the get_keys() function.
    ///
    pub fn get_titles(&self) -> Vec<String>
    {
        let mut titles = Vec::new();
        for item in self.json.values() {
            let item = item.as_object().unwrap();
            if !item.contains_key("percent") {
                continue;
            }
            let title = match item.get("title") {
                Option::None => "[unnamed]",
                Option::Some(val) => val.as_str().unwrap()
            };
            titles.push(String::from(title));
        }
        return titles;
    }
    
    ///
    /// Collects a (floating point) property across all assets within the portfolio.
    /// The returned vector aligns with the keys returned by the keys() function.
    ///
    pub fn get_mapped_property(&mut self, property_name : &str) -> Vec<f32>
    {
        let mut values = Vec::new();
        for item in self.json.values() {
            let item = item.as_object().unwrap();
            if !item.contains_key("percent") {
                continue;
            }
            let value = match item.get(property_name) {
                Option::None => 0.0,
                Option::Some(val) => val.as_f64().unwrap() as f32
            };
            values.push(value);
        }
        return values;
    }
    
    ///
    /// Sets a (floating point) property across all assets within the portfolio.
    /// The given values must be aligned with the keys returned by the get_keys() function.
    ///
    pub fn set_mapped_property(&mut self, property_name : &str, values : &Vec<f32>)
    {
        let mut idx = 0;
        for item in self.json.values_mut() {
            let mut item = item.as_object_mut().unwrap();
            if !item.contains_key("percent") {
                continue;
            }
            let new_value = JsonValue::from(((values[idx] as f64) * 10.0).round() / 10.0);
            if item.contains_key(property_name) {
                let mut property = item.get_mut(property_name).unwrap();
                *property = new_value;
            } else {
                item.insert(String::from(property_name), new_value);
            }
            idx += 1;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// Represents a period of chart data for an asset, in the form of a vector of equidistant values.
///
#[derive(Debug, Clone)]
pub struct ChartData
{
    pub name: String,
    pub values: Vec<f32>,
    start_timestamp: i64,
    end_timestamp: i64,
    interval: i64
}

#[allow(dead_code)]
impl ChartData
{
    ///
    /// Loads chart data for an asset from the conventional location in the data directory.
    ///
    pub fn load(name : &String, from_time : i64, to_time : i64) -> Self
    {
        let mut file_name = String::from(DATA_DIRECTORY);
        file_name.push_str(&name);
        file_name.push_str(EXTENSION_JSON);
        
        let mut file = File::open(file_name).unwrap();
        let mut data = String::new();
        file.read_to_string(&mut data).unwrap();
        let json : JsonValue = serde_json::from_str(&data).unwrap();
        
        let mut values : Vec<f32> = Vec::new();
        let mut start_timestamp : i64 = 0;
        let mut end_timestamp : i64 = 0;
        let mut interval : i64 = 0;
        
        let mut prev_time : i64 = 0;
        
        for element in json.as_array().unwrap().iter() {
            let element = element.as_array().unwrap();
            let time = element.get(0).unwrap().as_i64().unwrap();
            
            if (time < from_time) || (time > to_time) {
                continue;
            }
            
            end_timestamp = time;
            if start_timestamp == 0 {
                start_timestamp = time;
            }
            
            if interval != 0 {
                if time - prev_time != interval {
                    panic!("Intervals in data set {} are not equidistant.", name);
                }
            } else {
                if prev_time != 0 {
                    interval = time - prev_time;
                }
            }
            
            values.push(element.get(1).unwrap().as_f64().unwrap() as f32);
            prev_time = time;
        }
        
        if start_timestamp == 0 {
            panic!("No data points of {} in selected interval.", name);
        }
        
        return ChartData {
            name : name.clone(),
            values,
            start_timestamp,
            end_timestamp,
            interval
        };
    }
    
    ///
    /// Returns the unix timestamp (in milliseconds) of the first data value.
    ///
    pub fn get_start_timestamp(&self) -> i64
    {
        return self.start_timestamp;
    }

    ///
    /// Returns the unix timestamp (in milliseconds) of the last data value.
    ///
    pub fn get_end_timestamp(&self) -> i64
    {
        return self.end_timestamp;
    }
    
    ///
    /// Removes all values outside of the given timestamp boundaries.
    ///
    fn truncate(&mut self, from_time : i64, to_time : i64)
    {
        let mut crop_left : usize = 0;
        let mut new_len : usize = 0;
        let mut new_start : i64 = self.start_timestamp;
        let mut new_end : i64 = self.start_timestamp;
        
        for i in 0..self.values.len() {
            let time = self.start_timestamp + self.interval * (i as i64);
            if time < from_time {
                crop_left += 1;
                new_start = time + self.interval;
            } else if time <= to_time {
                new_len += 1;
                new_end = time;
            }
        }
        
        if crop_left > 0 {
            self.values.rotate_left(crop_left);
            self.start_timestamp = new_start;
            //println!("Cropped {} by {}", self.name, crop_left);
        } else {
            //println!("Data set {} caused other sets to be cropped.", self.name);
        }
        if new_len < self.values.len() {
            self.values.truncate(new_len);
            self.end_timestamp = new_end;
            //println!("Truncated {} to {}", self.name, new_len);
        }
    }
    
    ///
    /// Returns the SIMD iterable length (divisible by 8).
    ///
    fn simd_len(&self) -> usize
    {
        let mut len = self.values.len();
        let simd_offset = len % 8;
        if simd_offset != 0 {
            len += 8 - simd_offset;
        }
        return len;
    }
    
    ///
    /// Ensures that the number of allocated elemtents can be devided by 8.
    /// Space behind the last element is padded with 0.0 if necessary.
    ///
    fn ensure_simd_padding(&mut self)
    {
        let true_len = self.values.len();
        let simd_len = self.simd_len();
        for _ in 0..(simd_len - true_len) {
            self.values.push(0.0);            
        }
        self.values.truncate(true_len);
    }
    
    ///
    /// Will panic if the given other chart data does not contain data points
    /// which exactly align with ours (values of same index correspond to same time).
    ///
    fn assert_aligned_with(&self, other : &Self)
    {
        if (self.start_timestamp != other.start_timestamp) ||
           (self.end_timestamp != other.end_timestamp) ||
           (self.interval != other.interval)
        {
            panic!("Data points in {} do not align with {}.", self.name, other.name);
        }
    }
}

///
/// Represents multiple charts which are aligned (contain equal amounts of data points
/// with equal time intervals, so that same indices always correspond to the same time).
///
#[derive(Debug, Clone)]
pub struct AlignedChartDataSet
{
    pub charts: Vec<ChartData>
}

#[allow(dead_code)]
impl AlignedChartDataSet
{
    ///
    /// Loads multiple charts at once and ensures they are aligned.
    ///
    pub fn load(names: &[String], from_time : i64, to_time : i64) -> AlignedChartDataSet
    {
        let mut charts = Vec::new();
        let mut max_start_time = i64::min_value();
        let mut min_end_time = i64::max_value();
        for name in names.iter() {
            let chart = ChartData::load(name, from_time, to_time);
            max_start_time = max_start_time.max(chart.start_timestamp);
            min_end_time = min_end_time.min(chart.end_timestamp);
            charts.push(chart);
        }
        
        for chart in charts.iter_mut() {
            chart.truncate(max_start_time, min_end_time);
            chart.ensure_simd_padding();
        }
        for chart in charts.iter() {
            chart.assert_aligned_with(&charts[0]);
        }
        
        return AlignedChartDataSet{ charts };
    }
    
    ///
    /// Creates a linear combination of all contained charts.
    /// Each chart is scaled by the corresponding coefficient.
    ///
    pub fn combine(&self, coefficients: &[f32]) -> Vec<f32>
    {
        let coeff_map : Vec<(usize, f32)> = coefficients.iter()
            .map(|x| *x)
            .enumerate()
            .filter(|x| x.1 != 0.0)
            .collect();
        
        let simd_len = self.charts[0].simd_len();
        let mut combined = Vec::new();
        combined.reserve_exact(simd_len);
        
        unsafe {
            combined.set_len(self.chart_len());
            for i in (0..self.chart_len()).step_by(8) {
                let mut combined_values = _mm256_setzero_ps();
                for (j, coeff) in coeff_map.iter() {
                    let mut values = _mm256_loadu_ps(&self.charts[*j].values[i]);
                    let coeffs = _mm256_broadcast_ss(&coeff);
                    values = _mm256_mul_ps(values, coeffs);
                    combined_values = _mm256_add_ps(combined_values, values);
                }
                //Note: We are using the unaligned load/store instructions, since they are not
                //slower if the memory turns out to be aligned, but this is more compatible.
                //To ensure aligend memory, replace the global memory allocator.
                _mm256_storeu_ps(&mut combined[i], combined_values);
            }
        }
        
        return combined;
    }
    
    #[inline]
    pub fn len(&self) -> usize
    {
        return self.charts.len();
    }
    
    #[inline]
    pub fn chart_len(&self) -> usize
    {
        return self.charts[0].values.len();
    }
    
    ///
    /// Returns the unix timestamp (in milliseconds) of each first data value.
    ///
    pub fn get_start_timestamp(&self) -> i64
    {
        return self.charts[0].start_timestamp;
    }
    
    ///
    /// Returns the unix timestamp (in milliseconds) of each last data value.
    ///
    pub fn get_end_timestamp(&self) -> i64
    {
        return self.charts[0].end_timestamp;
    }
}
