using LightGraphs
using Printf
using DataFrames
using CSV
using GraphPlot

"""
    write_gph(dag::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s, %s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

# Bayes nets.jl
function compute(infile, outfile)
    println("Computing infile to outfile")
    # Read the data available
    data=CSV.File(infile) |> DataFrame;
    n= length(data); # i from 1 to n number of variables
    # Guess a graph
    G0= DiGraph(n);
    # compute score of current graph
    BayesianScore=calculateBayesianScore(data,G0)
    someConditionOnScore=false;
    numberOfAttempts=0;
    while (someConditionOnScore=false && numberOfAttempts<=100)
        # TODO make changes to the graph
        # TODO recompute score of new graph
        # increment numberOfAttempts
        numberOfAttempts+=1;
    end
    # write the latest graph to the outfile
    write_gph(dag::DiGraph, idx2names, filename)
end

function calculateBayesianScore(df::DataFrame, G::DiGraph)
    println("Calculating Bayesian score for a particular graph")
    n= length(df); # i from 1 to n number of variables
    r= describe(df).max; # index i has number of possible values ri for each variable
    parents=findParents(df,G); # parents[i] has the indices of parents of i
    q=zeros(Int64, n); # q[i] has number of parental instantiations qi
    M=Array{Any}(undef, n);
    for i = 1: n
        # Find q (number of parental instantiations) for each variable i
        if (length(parents[i])==0)
            q[i]=1;
        else
            for p in parents[i]
                q[i]+=r[p];
            end
        end
        # instantiate M with guess of prior which is uniform
        M[i]=zeros(Int64, q[i], r[i]);
        # access an element by looking at M[i][j,k]
    end
    for i=1: n
        parentIndices=parents[i];
        # aggregate data by itself and its parents
        aggs= by(df,vcat(i,parentIndices),nrow);
        for row in eachrow(aggs)
            # i= current variable=i
            # j=parental instantiation index for combination(row[2:end-1])
            parentValues=row[2:end-1];
            if (length(parentValues) ==0)
                j=1;
            else
                parentValues=convert(Array, parentValues)
                j=calculateParentalInstantiationIndex(parentIndices, parentValues, r);
            end
            k=row[1]; # Actual value of the variable i (first column of aggregation)
            incrementM=row[end]; # Mijk increment = row[end] (last column of aggregation)
            M[i][j,k]+=incrementM;
        end
    end
    return M
    # TODO Find score as a result of the M values
    # TODO return score
end

function findParents(df::DataFrame, G::DiGraph)
    n= length(df);
    # parents=Array{Array{Int64},1}(undef,n);
    # fill!(parents,[]); DONT DO THIS. IT FILLS WITH REFERENCE INSTEAD OF VALUE.
    parents=[Int64[] for i=1:n]
    edgesOfG=collect(edges(G));
    # TODO Look through edges of graph and find all the parents
    for e in edgesOfG
        par, chi= src(e), dst(e);
        push!(parents[chi],par);
    end
    return parents;
end

function calculateParentalInstantiationIndex(parentIndices, parentValues, r)
    r_parents=r[parentIndices]; # Number of instantiations of each parent
    A=fill(1,Tuple(r_parents)); # Array with dimensions corresponding to each parent
    linearIndices=LinearIndices(A);
    return linearIndices[parentValues...] # Using splat operator to index into linearIndices
end

# if length(ARGS) != 2
#     error("usage: julia project1.jl <infile>.csv <outfile>.gph")
# end

# inputfilename = ARGS[1]
# outputfilename = ARGS[2]

# inputfilename = "small.csv"
# outputfilename= "smallout.csv"
#
# compute(inputfilename, outputfilename)
data=CSV.File("myownsmallexample.csv") |> DataFrame;
n= length(data); # i from 1 to n number of variables
# Guess a undirected graph
G= DiGraph(n);
gplot(G, nodelabel=1:n)
add_edge!(G, 1, 2);
# compute score of current graph
M=calculateBayesianScore(data,G)
