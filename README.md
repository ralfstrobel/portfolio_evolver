# Portfolio Evolver - Investment via Genetic Algorithm

This command line application assists in the creation of optimally weighted investment portfolios,
picking assets based on performance contribution as well as low correlation.

Typical approaches to this problem only evaluate assets individually or in pairs (e.g. in the form of a correlation matrix),
which does not necessarily allow conclusions about their composit performance as a portfolio.
Therefore, this application chooses a more brute-force approach, simulating and assessing entire portfolios
by exploratively recombining asset charts, guided by a [genetic algorithm](http://en.wikipedia.org/wiki/Genetic_algorithm).

The project was originally conceived to evaluate social trading portfolios [on wikifolio.com](https://www.wikifolio.com/de/de/p/traderanalyst)
but can potentially be applied to any chart data.

## Setup

To build the application, an [installed Rust compiler](https://www.rust-lang.org/tools/install) is required.

The relative directory structure is currently hardcoded and should look like this...
* [repository] (git checkout location, e.g. "evolver")
  * src (rust source code)
  * target (compiled executables)
* "data"
  * Individual JSON chart data files, named like the asset id (e.g. "myISIN.json")
* "portfolios"
  * Individual JSON portfolio files that serve as both input and output
  * [current year] (e.g. "2021")
    * Copies of portfolio files, suffixed by date (output only, written after each run for archival purpose)
* "run" (recommended)
  * run scripts for specific portfolios

##### Chart Data Files

Chart data files describe a time series of asset values as a JSON array.
Each data point is itself an array of two values, a unix timestamp in milliseconds and a float value.

    [
        [1325462400000, 6075.52],
        [1325548800000, 6166.57],
        ...
    ]

This format is compatible with the [Highcharts](http://www.highcharts.com/) JavaScript library for easy chart display.

##### Portfolio Files

Portfolios are also stored in JSON format, as an object of objects.
The first object level uses asset ids as keys. The second level contains properties regarding the asset.

    {
        "US78378X1072": {
            "title": "S&P 500",
            "max_percent": 30.0,
            "percent": 10.0,
            "percent_max_uni_perf": 15.0,
            "percent_min_loss_area_2020": 5.0
        },
        ...
    }

Each asset has certain properties that are required but may contain any number of arbitrary properties...
* "title" - A human readable name for display in output summaries (string, required).
* "max_percent" - The maximum allocation weight simulations will assign to this asset (float, optional).
* "percent" - The current effective allocation weight (float, required - asset ignored otherwise).
* "percent_[objective][suffix]" - Output of a simulation as described below (float).

## Execution

Each run of the application executes a single simulation for a specified portfolio and objective function (see below).
The result will be written as a new property to each asset in the portfolio file, as well as a text summary file.

All simulation parameters are passed via environment variables. This facilitates the easy creation of run scripts
for specific simulation setups or batch execution of multiple related simulations.
* "PORTFOLIO" - The name of the portfolio file to use (without extension)
* "OPTIMIZATION_TARGET" - The objective function to use for the genetic algorithm
  * "max_performance" - Aims to maximize all time high portfolio value
  * "max_uni" - Aims to maximize the geometric uniformity of the portfolio chart (punishes volatility)
  * "max_uni_perf" - Aims to maximize both performance and uniformity (punishes downside volatility)
  * "max_gain_sum" - Aims to maximize the sum of all portfolio value gains on a relative logarithmic scale
  * "max_gain_length" - Aims to maximize the duration of portfolio value uptrends (punishes frequent losses)
  * "min_loss" - Aims to minimize the largest draw down of portfolio value
  * "min_loss_sum" - Aims to minimize the sum of all portfolio value drops on a relative logarithmic scale
  * "min_loss_length" - Aims to minimize the duration of portfolio value downtrends
  * "min_loss_area" - Aims to minimize the time integral of value loss during draw downs (depth x length)
* "OPTIMIZATION_NAME_SUFFIX" - Optional suffix to append to the result property when writing to portfolio
* "DATE_FROM" - Ignores older chart data (in case data files begin at different dates), format YYYY-MM-DD
* "DATE_TO" - Ignores newer chart data
* "GENE_MAX" - Defines a default for the "max_percent" property of all assets (as decimal number, not as percent) 
* "SAVE_RESULTS" - if set to "1", only displays output but does not write it

In addition to these standard parameters, additional variables exist to tweak the internals of the simulation:
* "NUM_SPECIES" - How many simulation runs to perform and average (more will stabilize results)
* "NUM_INDIVIDUALS_PER_GENE" - How many randomized portfolios to generate in relation to number of assets (less will increases performance, worsen results)
* "RAYON_NUM_THREADS" - How many simulation runs to execute in parallel (defaults to CPU cores - 1)

##### Example Run Script

    #!/bin/bash
    
    export PORTFOLIO=my_portfolio
    export GENE_MAX=0.075
    export NUM_INDIVIDUALS_PER_GENE=20
    export NUM_SPECIES=14
    
    export DATE_FROM=2016-01-01
    export DATE_TO=2020-12-31
    export OPTIMIZATION_NAME_SUFFIX=_2016_2020
    
    cd "$(dirname "$0")/.."
    OPTIMIZATION_TARGET=min_loss_area cargo run --release
    OPTIMIZATION_TARGET=max_uni_perf cargo run --release
