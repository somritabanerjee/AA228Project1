using LightGraphs
using Printf
using DataFrames
using CSV
using GraphPlot
using SpecialFunctions
using BayesNets # only for testing my scoring function against bayesian_score

# """
#     ScoringParams
# This struct stores the parameters that are required for the scoring function.
# """
struct ScoringParams
    n::Int64
    q::Vector{Int64}
    r::Vector{Int64}
    parents
    M
end

# """
#     ResultsOfAddingParents
# This struct stores the results of adding a parent p when choosing between
#     multiple nodes to be added as parents.
# """
struct ResultsOfAddingParents
    parents::Vector{Int64}
    scores::Vector{Float64}
    scoringParams::Vector{ScoringParams}
    graphs::Vector{DiGraph}
    ResultsOfAddingParents(w,x,y,z) = size(w)!=size(x) || size(x) != size(y) || size(y)!=size(z) ? error("Size of four vectors must be the same") : new(w,x,y,z)
end

# """
#     write_gph(dag::DiGraph, idx2names, filename)
# Takes a DiGraph, a Dict of index to names and a output filename to write the
# graph in `gph` format.
# """
function write_gph(dag::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(dag)
            @printf(io, "%s, %s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

# """
#     compute(infile::String, outfile::String)
# Takes an input and output file name and reads the data, performs a K2 search,
# and outputs the graph with the highest Bayesian score.
# """
function compute(infile::String, outfile::String)
    println("Computing infile to outfile")
    # Read the data available
    if (infile=="large.csv")
        # CSV.File read takes too long. Using readtable (deprecated) instead.
        data = readtable("large.csv", separator=',', header=true);
    else
        data=CSV.File(infile) |> DataFrame;
    end
    n= length(data); # number of variables
    # Create dictionary from index to names
    idx2names= Dict(zip(collect(1:n), names(data)))
    # Perform K2 Search
    if (infile=="large.csv")
        (bestScore, scoringParams, newG)= performK2SearchWithPrior(data, maxNumberOfAttempts=300)
    else
        (bestScore, scoringParams, newG)= performK2Search(data, maxNumberOfAttempts=100)
    end
    # Write the graph to the outfile
    write_gph(newG::DiGraph, idx2names, outfile)
end

# """
#     performK2Search(data::DataFrame; maxNumberOfAttempts=20)
# Takes the data as well as an optional parameter that specifies the max number of
# node additions that should be attempted. The default value for this is 20.
# Performs a K2 search starting with a completely unconnected graph to find the
# graph that maximizes the score at each step.
# """
function performK2Search(data::DataFrame; maxNumberOfAttempts=20)
    n= length(data);
    # Guess a graph
    G= DiGraph(n);
    # Compute score of current graph
    scoringParams=findScoringParams(data,G);
    score=calculateBayesianScore(scoringParams)
    searchComplete=false;
    numberOfAttempts=0;
    while (!searchComplete && numberOfAttempts<maxNumberOfAttempts)
        # Make changes to the graph
        (newScore, newScoringParams, newG, searchComplete)=
         makeSingleChangeToGraph(G, data, scoringParams, score, searchComplete);
        score=newScore;
        scoringParams=newScoringParams;
        G=newG;
        # Increment numberOfAttempts
        numberOfAttempts+=1;
    end
    return (score, scoringParams, G)
end

function performK2SearchWithPrior(data::DataFrame; maxNumberOfAttempts=20)
    n= length(data);
    # Guess a graph
    G= DiGraph(n);
    add_edge!(G,6,1);
    add_edge!(G,2,1);
    add_edge!(G,49,1);
    add_edge!(G,7,1);
    add_edge!(G,22,1);
    add_edge!(G,21,1);
    add_edge!(G,13,1);
    add_edge!(G,10,1);
    add_edge!(G,33,2);
    add_edge!(G,7,2);
    add_edge!(G,14,2);
    add_edge!(G,18,2);
    add_edge!(G,8,2);
    add_edge!(G,15,2);
    add_edge!(G,4,3);
    add_edge!(G,5,3);
    add_edge!(G,11,3);
    add_edge!(G,10,3);
    add_edge!(G,5,4);
    add_edge!(G,26,4);
    add_edge!(G,49,4);
    add_edge!(G,23,4);
    add_edge!(G,11,4);
    add_edge!(G,48,4);
    add_edge!(G,10,4);
    add_edge!(G,26,5);
    add_edge!(G,49,5);
    add_edge!(G,23,5);
    add_edge!(G,11,5);
    add_edge!(G,10,5);
    add_edge!(G,9,5);
    add_edge!(G,8,5);
    add_edge!(G,23,6);
    add_edge!(G,37,6);
    add_edge!(G,20,6);
    add_edge!(G,36,6);
    add_edge!(G,39,6);
    add_edge!(G,21,6);
    add_edge!(G,13,6);
    add_edge!(G,10,6);
    add_edge!(G,8,7);
    add_edge!(G,9,7);
    add_edge!(G,14,7);
    add_edge!(G,41,7);
    add_edge!(G,6,7);
    add_edge!(G,9,8);
    add_edge!(G,20,8);
    add_edge!(G,10,8);
    add_edge!(G,14,8);
    add_edge!(G,15,8);
    add_edge!(G,18,8);
    add_edge!(G,20,9);
    add_edge!(G,10,9);
    add_edge!(G,44,9);
    add_edge!(G,12,9);
    add_edge!(G,41,9);
    add_edge!(G,50,9);
    add_edge!(G,12, 10)
    add_edge!(G,21, 10)
    add_edge!(G,11, 10)
    add_edge!(G,38, 10)
    add_edge!(G,23, 10)
    add_edge!(G,12, 11)
    add_edge!(G,26, 11)
    add_edge!(G,25, 11)
    add_edge!(G,23, 11)
    add_edge!(G,38, 11)
    add_edge!(G,21, 12)
    add_edge!(G,29, 12)
    add_edge!(G,44, 12)
    add_edge!(G,41, 12)
    add_edge!(G,24, 12)
    add_edge!(G,13, 12)
    add_edge!(G,28, 12)
    add_edge!(G,16, 13)
    add_edge!(G,15, 13)
    add_edge!(G,22, 13)
    add_edge!(G,38, 13)
    add_edge!(G,15, 14)
    add_edge!(G,19, 14)
    add_edge!(G,9, 14)
    add_edge!(G,41,14)
    add_edge!(G,20,14)
    add_edge!(G,22,14)
    add_edge!(G,21,14)
    add_edge!(G,17,15)
    add_edge!(G,25,15)
    add_edge!(G,24,15)
    add_edge!(G,22,15)
    add_edge!(G,41,15)
    add_edge!(G,15,16)
    add_edge!(G,29,16)
    add_edge!(G,38,16)
    add_edge!(G,21,16)
    add_edge!(G,28,16)
    add_edge!(G,25,17)
    add_edge!(G,28,17)
    add_edge!(G,24,17)
    add_edge!(G,22,17)
    # Compute score of current graph
    scoringParams=findScoringParams(data,G);
    score=calculateBayesianScore(scoringParams)
    searchComplete=false;
    numberOfAttempts=0;
    while (!searchComplete && numberOfAttempts<maxNumberOfAttempts)
        # Make changes to the graph
        (newScore, newScoringParams, newG, searchComplete)=
         makeSingleChangeToGraph(G, data, scoringParams, score, searchComplete);
        score=newScore;
        scoringParams=newScoringParams;
        G=newG;
        # Increment numberOfAttempts
        numberOfAttempts+=1;
    end
    return (score, scoringParams, G)
end

# """
#     makeSingleChangeToGraph(G::DiGraph, data::DataFrame, oldScoringParams::ScoringParams, oldScore::Float64, searchComplete::Bool)
# Takes in the graph, data, previous scoring parameters, previous score, and a
# bool. Starting from node 1, tries to find some other node p such that adding an
# edge from p to 1 increases the Bayesian score. Picks the edge p-> 1 that
# maximizes the Bayesian score. If not possible to draw an edge to 1 and increase
# score, it moves on to the next node 2. If all n nodes have been tried, returns
# searchComplete as true.
# """
function makeSingleChangeToGraph(G::DiGraph, data::DataFrame, oldScoringParams::ScoringParams, oldScore::Float64, searchComplete::Bool)
    n=oldScoringParams.n;
    parents=oldScoringParams.parents;
    currentNodeIndex = 1;
    while currentNodeIndex <= n
        resultsOfAddingParents= ResultsOfAddingParents([], [], ScoringParams[], DiGraph[]);
        for p=1:n
            newG=copy(G);
            add_edge!(newG, p, currentNodeIndex);
            # If this p is the current node or already a parent of the current
            # node or it makes graph cyclic, skip it
            if (p==currentNodeIndex) || (p in parents[currentNodeIndex] || is_cyclic(newG))
                continue;
            end
            # Otherwise try scoring by adding p as a parent of currentNodeIndex
            newScoringParams=updateScoringParams(data, oldScoringParams, currentNodeIndex, p);
            newScore=calculateBayesianScore(newScoringParams);
            if newScore>oldScore
                push!(resultsOfAddingParents.parents, p);
                push!(resultsOfAddingParents.scores, newScore);
                push!(resultsOfAddingParents.scoringParams, newScoringParams);
                push!(resultsOfAddingParents.graphs, newG);
            end
        end
        # Return the best score if something is found
        if !isempty(resultsOfAddingParents.parents)
            (newScore, idx) = findmax(resultsOfAddingParents.scores);
            parentAdded = resultsOfAddingParents.parents[idx];
            println("    Adding parent $parentAdded to index $currentNodeIndex to increase score to $newScore")
            newScoringParams = resultsOfAddingParents.scoringParams[idx];
            newG = resultsOfAddingParents.graphs[idx];
            return (newScore, newScoringParams, newG, searchComplete);
        end
        # Otherwise, move on to next node
        currentNodeIndex+=1;
    end
    # Gone through all the nodes and no better graph was found
    searchComplete = true;
    println("Search terminated because no more nodes left")
    # Return old score and scoring params
    return (oldScore, oldScoringParams, G, searchComplete);
end

# """
#     findScoringParams(df::DataFrame, G::DiGraph)
# For the first time the Bayesian score is being computed for a dataset on a
# graph, we need to find all the scoring parameters from scratch. This means that
# we find all the r,q, and Mijk values and return them in a struct ScoringParams.
# """
function findScoringParams(df::DataFrame, G::DiGraph)
    n= length(df); # i from 1 to n number of variables
    r= describe(df).max; # index i has number of possible values ri for each variable
    parents=findParents(df,G); # parents[i] has the indices of parents of i
    q=ones(Int64, n); # q[i] has number of parental instantiations qi
    M=Array{Any}(undef, n);
    for i = 1: n
        # Find q (number of parental instantiations) for each variable i
        for p in parents[i]
            q[i]*=r[p];
        end
        # Instantiate M with guess of prior which is uniform
        M[i]=zeros(Int64, q[i], r[i]);
        # Access an element by looking at M[i][j,k]
    end
    for i=1: n
        parentIndices=parents[i];
        # Aggregate data by itself and its parents
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
    return ScoringParams(n,q,r,parents,M);
end

# """
#     updateScoringParams(df::DataFrame, oldParams::ScoringParams, i_node::Int64, i_parent::Int64)
# After the scoring parameters have been calculated once, each time we add an edge
# to the graph we simply need to recalculate the Mijk for that particular i for
# which a parent has been added. This function updates the scoring parameters and
# returns a new struct of the type ScoringParams.
# """
function updateScoringParams(df::DataFrame, oldParams::ScoringParams, i_node::Int64, i_parent::Int64)
    # Copying scoringParams into a new variable
    params=deepcopy(oldParams);
    n=params.n; # doesn't change
    r=params.r; # doesn't change
    q=params.q; # only q[i_node] changes
    parents=params.parents; # only parents[i_node] changes
    M=params.M; # only M[i_node] changes
    # Changing parents
    push!(parents[i_node], i_parent);
    # Changing q
    q[i_node]*=r[i_parent];
    # Changing M
    M[i_node]=zeros(Int64, q[i_node], r[i_node]);
    parentIndices=parents[i_node];
    # Aggregate data by itself and its parents
    aggs= by(df,vcat(i_node,parentIndices),nrow);
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
        M[i_node][j,k]+=incrementM;
    end
    return ScoringParams(n,q,r,parents,M);
end

# """
#     calculateBayesianScore(params::ScoringParams)
# Given the scoring parameters n, q, r, M_ijk, we can calculate the logarithm
# of the Bayesian score by using the log gamma function. This function returns
# the float value of this log(Bayesian score).
# """
function calculateBayesianScore(params::ScoringParams)
    n=params.n;
    q=params.q;
    r=params.r;
    M=params.M;
    # Find score as a result of the M values
    score=0;
    for i=1: n
        for j=1:q[i]
            alpha_ij0=1*r[i];
            M_ij0=sum(M[i][j,:])
            score+=lgamma(alpha_ij0)-lgamma(alpha_ij0+M_ij0);
            for k=1:r[i]
                alpha_ijk=1;
                score+=lgamma(alpha_ijk +M[i][j,k])-lgamma(alpha_ijk);
            end
        end
    end
    return score
end

# '''
#     findParents(df::DataFrame, G::DiGraph)
# This function finds all the parents of each node i in the graph G and returns
# a vector of size n where parents[i] is a vector containing the indices of the
# parents of i.
# '''
function findParents(df::DataFrame, G::DiGraph)
    # Look through edges of graph and find all the parents
    n= length(df);
    parents=[Int64[] for i=1:n]
    for i=1:n
        parents[i]=inneighbors(G, i);
    end
    # This implementation MIGHT be faster DONT DELETE
    # edgesOfG=collect(edges(G));
    # for e in edgesOfG
    #     par, chi= src(e), dst(e);
    #     push!(parents[chi],par);
    # end
    return parents;
end

# '''
#     calculateParentalInstantiationIndex(parentIndices, parentValues, r)
# This function calculates the j value for a given instantiation of the parents of
# i given all possible values of each parent (encoded in r). This makes use of
# the LinearIndices function.
# '''
function calculateParentalInstantiationIndex(parentIndices, parentValues, r)
    r_parents=r[parentIndices]; # Number of instantiations of each parent
    A=fill(1,Tuple(r_parents)); # Array with dimensions corresponding to each parent
    linearIndices=LinearIndices(A);
    return linearIndices[parentValues...] # Using splat operator to index into linearIndices
end

# '''
#     createAndRunTestsForScoringFunction
# This function creates a fairly large number of test cases for me to test my
# implementation of the Bayesian score function and compares it against the
# bayesian_score function in BayesNet.jl. NOTE: I did not use bayesian_score
# for anything else except testing.
# '''
function createAndRunTestsForScoringFunction()
    data=CSV.File("myownsmallexample.csv") |> DataFrame;
    n= length(data); # i = number of variables
    G= DiGraph(n);
    G1= DiGraph(n);
    add_edge!(G1, 1, 2);
    testScoringFunction(data,G,testCase=1)
    testScoringFunction(data,G1,testCase=2)
    data=CSV.File("small.csv") |> DataFrame;
    n= length(data); # i = number of variables
    G= DiGraph(n);
    G1= DiGraph(n);
    add_edge!(G1, 1, 2);
    G2=copy(G1);
    add_edge!(G2, 3, 2);
    testScoringFunction(data,G,testCase=3,printingOn=true)
    scParamsForG1=testScoringFunction(data,G1,testCase=4,printingOn=true)
    scParamsForG2=testScoringFunction(data,G2,testCase=5,printingOn=true)
    testUpdatingScoringFunction(data, G2, scParamsForG1, 2, 3, testCase=6, printingOn=true)
    G3=copy(G2);
    add_edge!(G3, 4, 2);
    testUpdatingScoringFunction(data, G3, scParamsForG2, 2, 4, testCase=7, printingOn=true)
    data=CSV.File("medium.csv") |> DataFrame;
    n= length(data); # i = number of variables
    G= DiGraph(n);
    G1= DiGraph(n);
    add_edge!(G1, 1, 2);
    scForG=testScoringFunction(data,G,testCase=8,printingOn=true)
    testScoringFunction(data,G1,testCase=9,printingOn=true)
    testUpdatingScoringFunction(data, G1, scForG, 2, 1, testCase=10, printingOn=true)
    G2=copy(G);
    add_edge!(G2,9,1);
    testUpdatingScoringFunction(data, G2, scForG, 1, 9, testCase=11, printingOn=true)
    G3=copy(G);
    add_edge!(G3,10,1);
    testUpdatingScoringFunction(data, G3, scForG, 1, 10, testCase=12, printingOn=true)
    # This read takes too long. Using readtable (deprecated) instead.
    # data=CSV.File("large.csv") |> DataFrame;
    data_large = readtable("large.csv", separator=',', header=true)
    n= length(data); # i = number of variables
    G= DiGraph(n);
    G1= DiGraph(n);
    add_edge!(G1, 1, 2);
    # TODO Currently these take too long to run. Fix scoring function.
    scForG=testScoringFunction(data,G,testCase=13,printingOn=true)
    testUpdatingScoringFunction(data, G1, scForG, 2, 1, testCase=14, printingOn=true)
end

# '''
#     testScoringFunction(data::DataFrame, G::DiGraph; testCase=0, printingOn=true)
# This function is a helper to test my scoring function against bayesian_score.
# '''
function testScoringFunction(data::DataFrame, G::DiGraph; testCase=0, printingOn=true)
    scoringParams=findScoringParams(data,G);
    myscore=calculateBayesianScore(scoringParams);
    theirscore=bayesian_score(G, names(data), data);
    if (printingOn)
        println("my score = $myscore")
        println("their score = $theirscore")
    end
    if (abs(myscore-theirscore)<10^-5)
        println("Test case $testCase passed!")
        println("");
    else
        println("ERROR for test case $testCase")
        println("");
    end
    return scoringParams
end

# '''
#     testUpdatingScoringFunction(data::DataFrame, G::DiGraph, oldParams::ScoringParams, i_node::Int64, i_parent::Int64; testCase=0, printingOn=true)
# This function is a helper to test my updating score function against bayesian_score.
# '''
function testUpdatingScoringFunction(data::DataFrame, G::DiGraph, oldParams::ScoringParams, i_node::Int64, i_parent::Int64; testCase=0, printingOn=true)
    scoringParams=updateScoringParams(data, oldParams, i_node, i_parent);
    myscore=calculateBayesianScore(scoringParams);
    theirscore=bayesian_score(G, names(data), data);
    if (printingOn)
        println("my score = $myscore")
        println("their score = $theirscore")
    end
    if (abs(myscore-theirscore)<10^-5)
        println("Test case $testCase passed!")
        println("");
    else
        println("ERROR for test case $testCase")
        println("");
    end
end

# '''
#     createAndRunTestsForK2Search()
# This function is used to create and run test cases to see if the K2 search
# performs well for different graphs. I only allow 20 steps for this basic case.
# '''
function createAndRunTestsForK2Search()
    data=CSV.File("myownsmallexample.csv") |> DataFrame;
    println("Trying to find best graph for myownsmallexample.csv");
    (score, scoringParams)=performK2Search(data::DataFrame; maxNumberOfAttempts=20);
    println("Score optimized= $score");
    println("");
    data_small=CSV.File("small.csv") |> DataFrame;
    println("Trying to find best graph for small.csv")
    (score, scoringParams)=performK2Search(data_small::DataFrame; maxNumberOfAttempts=20);
    println("Score optimized= $score");
    println("");
    data_medium=CSV.File("medium.csv") |> DataFrame;
    println("Trying to find best graph for medium.csv");
    (score, scoringParams)=performK2Search(data_medium::DataFrame; maxNumberOfAttempts=20);
    println("Score optimized= $score");
    println("");
    data_large = readtable("large.csv", separator=',', header=true)
    # data_large=CSV.File("large.csv") |> DataFrame;
    println("Trying to find best graph for large.csv");
    (score, scoringParams)=performK2Search(data_large::DataFrame; maxNumberOfAttempts=10);
    println("Score optimized= $score");
    println("");
end

# Run tests first
createAndRunTestsForScoringFunction()
createAndRunTestsForK2Search()

# Create the files for submission
inputfilename = ["small.csv", "medium.csv", "large.csv"]
outputfilename= ["small.gph", "medium.gph", "large.gph"]
for i=1:3
    compute(inputfilename[i], outputfilename[i])
end

# Congrats. You've reached the end! Go google puppy pictures now.
