# Hierarchical clustering solver

To use:

`python hcsolver.py [-v] [--depth <int>] [--linkage <linkagetype>] [--dist <distancefunction>] input-file`

The optional `-v` argument enables verbose output; if you do not use verbose mode, you will need to use this program interactively and inspect the created `hc` object (see source). Verbose mode is enabled by default. The optional `--depth` argument sets a clustering depth (number of clusters to form); if this is less than one, a full clustering will be performed. (If it is higher than the available number of clusters, no error will be thrown, but no calculations will be performed.) If no depth value is specified, it will be zero by default for full clustering. The optional `--linkage` argument sets the type of linkage to use for clustering; it may be one of `single` or `complete`. If no argument is specified, single linkage will be used by default. The optional `--dist` argument sets the distance metric to use; it may be one of `manh` for Manhattan distance or `e2` for Euclidean distance squared. If no argument is specified, Euclidean distance squared will be used by default. The `inputfile` must conform to the specification below.

**Regarding input files**

Input files should be in a comma-separated format, where each line is a string of comma-separated numeric values (floating point are okay), with the last column being an alphanumeric label (underscores also okay) for the row (i.e., initial cluster labels). Each row must contain the same number of comma-separated values or the program will exit with an error.

### Exit codes:

* `0`: normal exit
* `1`: insufficient number of arguments passed at command line
* `2`: bad argument for command line option
* `3`: unexpected error during argument parsing
* `4`: error during input file parsing
