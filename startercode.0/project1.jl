using LightGraphs
using Printf
using DataFrames
using CSV

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

function calculateParentalInstantiationIndex(parents_i, parentalValues, r)
    r_parents_i=r[parents_i]; # Number of instantiations of each parent
    A=fill(1,Tuple(r_parents_i)); # Array with dimensions corresponding to each parent
    linearIndices=LinearIndices(A);
    return linearIndices[parentalValues...] # Using splat operator to index into linearIndices
end

# EXTRA DELETE
# say the parents have [rA rB] possible values
# Ordering [1 1] [1 2] [1 3]... [1 rB] [2 1] [2 2] [2 3]... [2 rB]
# Therefore [i j]=rB*(i-1)+j. Similarly [i j k]=rB*rC*(i-1)+rC*(j-1)+k
# Bayes nets.jl
function compute(infile, outfile)
    # Read the data available
    df=CSV.File(infile) |> DataFrame;
    n= length(df); # i from 1 to n number of variables
    # Guess a graph
    G0= DiGraph(n);
    # compute score of current graph
    BayesianScore=calculateBayesianScore(G0)
    someConditionOnScore=false;
    numberOfAttempts=0;
    while (someConditionOnScore=false && numberOfAttempts<=100)
        # make changes to the graph
        # recompute score of new graph
        # increment numberOfAttempts
        numberOfAttempts ++;
    end
    # write the latest graph to the outfianSile
    write_gph(dag::DiGraph, idx2names, filename)
end

function calculateBayes
    n= length(df); # i from 1 to n number of variables
    r= describe(df).max; # index i has number of possible values ri for each variable
    parents=Array{Any}(undef, n); # parents[i] has the indices of parents of i
    q=ones(Int64, n); # q[i] has number of parental instantiations qi
    M=Array{Any}(undef, n);
    for i = 1: n
        M[i]=zeros(Int64, q[i], r[i]);
        # access an element by looking at M[i][j,k]
        # if i=1
        #     parents[i]=[3,4];
        # else
            parents[i]=Int64[];
        # end
        if (length(parents[i])==0)
            q[i]=1;
        else
            for p in parents[i]
                q[i]+=r[p];
            end
        end
    end
    for i=1: n
        parents_i=parents[i];
        # aggregate by itself and its parents
        aggs= by(df,vcat(i,parents_i),nrow);
        for row in eachrow(aggs)
            # i= current variable=i
            # j=parental instantiation index for combination(row[2:end-1])
            parentalValues=row[2:end-1];
            if (length(parentalValues) ==0)
                j=1;
            else
                j=calculateParentalInstantiationIndex(parents_i, parentalValues, r);
            end
            k=row[1]; # Actual value of the variable i
            incrementM=row[end]; # Mijk increment = row[end]
            M[i][j,k]+=incrementM;
        end
    end
    CSV.write(outfile, df)
    # WRITE YOUR CODE HERE
     # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

end

# if length(ARGS) != 2
#     error("usage: julia project1.jl <infile>.csv <outfile>.gph")
# end

# inputfilename = ARGS[1]
# outputfilename = ARGS[2]

inputfilename = "small.csv"
outputfilename= "smallout.csv"

compute(inputfilename, outputfilename)
