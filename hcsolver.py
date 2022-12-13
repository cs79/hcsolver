#===================#
#   I M P O R T S   #
#===================#

import sys
import getopt
import re
from collections import OrderedDict
from enum import Enum
import pandas as pd

#=========================#
#   S T R U C T U R E S   #
#=========================#

class HC():
    """
    Container object for hierarchical clustering data.
    """
    def __init__(self):
        self.clusters = OrderedDict()
        self.cmemberof = OrderedDict()
        self.distmatrix = pd.DataFrame()
        self.sorteddists = []
    def __repr__(self):
        rs = "Hierarchical Clustering object contents:\n"
        rs += "----------------------------------------\n\n"
        rs += "clusters:\n\n{}\n\n".format(self.clusters)
        rs += "cmemberof:\n\n{}\n\n".format(self.cmemberof)
        rs += "distmatrix:\n\n{}\n\n".format(self.distmatrix)
        rs += "sorteddists:\n\n{}\n\n".format(self.sorteddists)
        return rs

class Linkage(Enum):
    """
    Simple enumeration of linkage types.
    """
    SINGLE = 1
    COMPLETE = 2

class DistMetric(Enum):
    """
    Simple enumeration of possible distance metrics.
    """
    EUCLIDEANSQUARED = 1
    MANHATTAN = 2



#=====================================#
#   H E L P E R   F U N C T I O N S   #
#=====================================#

def usage():
    """
    Notifies user of how to run this program.
    """
    print("Usage: python hcsolver.py [-v] [--depth <int>] [--linkage <linkagetype>] [--dist <distancefunction>] input-file")

def parse_args():
    """
    Parses command-line arguments to potentially override default parameters.
    Exits with an error if invalid commands detected.
    Returns a dictionary containing final argument values otherwise.
    """
    # check that we have enough minimal arguments to not immediately crash
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)
    # set default values; check runtime arguments to see if we need overrides
    args = {}
    args["verbose"] = False
    args["depth"] = 0                               # < 1 indicates full clustering
    args["linkage"] = Linkage.SINGLE                # default to simpler case
    args["distance"] = DistMetric.EUCLIDEANSQUARED  # "reasonable" default
    # parse arguments and update args dict
    args["inputfile"] = sys.argv[-1]
    arg_list = sys.argv[1:-1]
    short_opt = "v"
    long_opt = ["depth=", "linkage=", "dist="]
    try:
        passed_args, _ = getopt.getopt(arg_list, short_opt, long_opt)
        for arg, val in passed_args:
            if arg == "-v":
                args["verbose"] = True
            if arg == "--depth":
                val = int(val)
                if val < 1:
                    print("Implied zero or fewer clusters to form; performing full clustering")
                else:
                    args["depth"] = val
            if arg == "--linkage":
                if val in ["S", "s", "single"]:
                    pass
                elif val in ["C", "c", "complete"]:
                    args["linkage"] = Linkage.COMPLETE
                else:
                    print("Linkage value was invalid; using single linkage for this run. Valid values are `single` or `complete`.")
            if arg == "--dist":
                if val not in ["manh", "e2"]:
                    print ("dist argument must be one of: manh, e2")
                    sys.exit(2)
                if val == "manh":
                    args["distance"] = DistMetric.MANHATTAN
                if val == "e2":
                    args["distance"] = DistMetric.EUCLIDEANSQUARED
    except getopt.error as err:
        print("Error during parsing of option arguments: {}".format(err))
        usage()
        sys.exit(3)
    return args

def parse_input_file(filename):
    """
    Reads input file one line at a time and parses into a list.
    All but the final value in the line must be numeric; the final
    value may be an alphanumeric identifier with underscores allowed.
    If not all rows have the same number of entries, or if any line
    does not conform to the input specification, the program will
    exit with an error.
    Returns the parsed set of rows as a dict of lists otherwise, using
    the final list entry per row as the dict key.
    """
    filedata = OrderedDict()
    # regex for line format
    pat_line = re.compile("^\d+(.\d+)(,\d+(.\d+))*,[a-zA-z0-9_]+")
    # open file and read line by line
    f = open(filename)
    lines = f.readlines()
    for line in lines:
        line = re.sub("\s", "", line)
        # skip blank lines
        if line == "":
            continue
        if pat_line.match(line) != None:
            this_line = line.split(",")
            try:
                for i in range(0, len(this_line) - 1):
                    this_line[i] = float(this_line[i])
                filedata[this_line[-1]] = this_line[:-1]
            except:
                "Unable to cast column values to numeric - check input file"
                sys.exit(4)
        else:
            print("Input lines must conform to specification in README")
            sys.exit(4)
    # check for column length consistency
    keys = list(filedata.keys())
    ncols = len(filedata[keys[0]])
    for key in keys:
        if len(filedata[key]) != ncols:
            print("All rows in input data files must have the same number of columns")
            sys.exit(4)
    return filedata

def get_initial_clusters(clist):
    """
    Sets up two objects to contain related cluster information.
    The `clusters` object is a dictionary mapping cluster names to
    cluster contents; the key is a formatted cluster name string,
    and the value is a flat list of original cluster labels belonging
    to the named cluster.
    The `cmemberof` object is an inverse lookup where the key is an
    original cluster label, and the value is the current name string
    of the cluster containing that label.
    Returns (`clusters`, `cmemberof`) as a packed tuple.
    """
    clusters = OrderedDict()
    cmemberof = OrderedDict()
    for c in clist:
        clusters[wrap1(c)] = [c]
        cmemberof[c] = wrap1(c)
    return clusters, cmemberof

def calc_distance_matrix(dists, distmetric):
    """
    Calculates distance matrix for all pointwise distances.
    The `dists` object is assumed to be an OrderedDict / dict,
    where keys represent (initial) clusters and values are (2D)
    lists locating those clusters.
    Returns a dataframe representing the distance matrix.
    """
    df = pd.DataFrame(columns=dists.keys(), index=dists.keys())
    for key1 in dists:
        for key2 in dists:
            if key1 == key2:
                # zero distance to self
                df[key1][key2] = 0
            else:
                dist = None
                if distmetric == DistMetric.EUCLIDEANSQUARED:
                    dist = calc_pointwise_e2(dists[key1], dists[key2])
                if distmetric == DistMetric.MANHATTAN:
                    dist = calc_pointwise_manh(dists[key1], dists[key2])
                assert dist is not None, "Invalid distance metric!"
                # set dist in matrix
                df[key1][key2] = dist
    return df

def calc_pointwise_e2(p1, p2):
    """
    Pointwise Euclidean distance squared between points `p1` and `p2`.
    Assumes points are represented in list form.
    Also assumes points are same dimensionality, or else this will crash.
    """
    totaldist = 0
    for i in range(len(p1)):
        totaldist += (p1[i] - p2[i]) ** 2
    return totaldist

def calc_pointwise_manh(p1, p2):
    """
    Pointwise Manhattan distance between points `p1` and `p2`.
    Assumes points are represented in list form.
    Also assumes points are same dimensionality, or else this will crash.
    """
    totaldist = 0
    for i in range(len(p1)):
        totaldist += abs(p1[i] - p2[i])
    return totaldist

def get_sorted_dists(distmatrix):
    """
    Extracts all distances from half of symmetric `distmatrix` DataFrame.
    Sorts these and returns them in a list.
    """
    sorteddists = []
    keys = list(distmatrix.columns)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            # print(keys[i], keys[j])
            sorteddists.append(distmatrix[keys[i]][keys[j]])
    sorteddists.sort()
    return sorteddists

def wrap1(nodestring):
    return "{" + nodestring + "}"

def wrap2(ns1, ns2):
    return "{" + ns1 + "," + ns2 + "}"

#=========================================#
#   C O R E   F U N C T I O N A L I T Y   #
#=========================================#

def run_hc(hc, linkage="single", verbose=False):
    """
    Runs hierarchical clustering on `hc.clusters` initial clusters.
    The `hc` object is assumed to be of type `HC`, and to have been
    populated with distance information to be used for clustering.
    """
    



#=================================================#
#   M A I N   P R O G R A M   E X E C U T I O N   #
#=================================================#

def main():
    # parse runtime arguments
    args = parse_args()
    # parse input file distance data into OrderedDict
    dists = parse_input_file(args["inputfile"])
    # extract / calculate various data and pack into HC object
    hc = HC()
    hc.clusters, hc.cmemberof = get_initial_clusters(list(dists.keys()))
    hc.distmatrix = calc_distance_matrix(dists, args["distance"])
    hc.sorteddists = get_sorted_dists(hc.distmatrix)

    # run clustering, printing out results
    run_hc(hc, args["linkage"], args["verbose"])

    # if no crashes so far, just exit normally
    sys.exit(0)

if __name__ == "__main__":
    main()
