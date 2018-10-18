using LightGraphs
using Printf
using DataFrames
using CSV
using GraphPlot
using SpecialFunctions
using BayesNets

struct ScoringParams
    n::Int64
    q
    r
    parents
    M
end

struct ResultsOfAddingParents
    parents::Vector{Int64}
    scores::Vector{Float64}
    scoringParams::Vector{ScoringParams}
    graphs::Vector{DiGraph}
    ResultsOfAddingParents(w,x,y,z) = size(w)!=size(x) || size(x) != size(y) || size(y)!=size(z) ? error("Size of four vectors must be the same") : new(w,x,y,z)
end

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

function compute(infile, outfile)
    println("Computing infile to outfile")
    # Read the data available
    data=CSV.File(infile) |> DataFrame;
    n= length(data); # number of variables
    # Create dictionary from index to names
    #  TODO remove nodes not in any edges
    idx2names= Dict(zip(collect(1:n), names(data)))
    # Perform K2 Search
    (bestScore, scoringParams, newG)= performK2Search(data, maxNumberOfAttempts=10)
    # write the graph to the outfile
    write_gph(newG::DiGraph, idx2names, outfile)
end

function performK2Search(data::DataFrame; maxNumberOfAttempts=10)
    n= length(data);
    # Guess a graph
    G= DiGraph(n);
    # compute score of current graph
    scoringParams=findScoringParams(data,G);
    score=calculateBayesianScore(scoringParams)
    searchComplete=false;
    numberOfAttempts=0;
    while (!searchComplete && numberOfAttempts<maxNumberOfAttempts)
        # Make changes to the graph
        (newScore, newScoringParams, newG, searchComplete)=makeSingleChangeToGraph(G, data, scoringParams, score, searchComplete);
        score=newScore;
        scoringParams=newScoringParams;
        G=newG;
        # increment numberOfAttempts
        numberOfAttempts+=1;
    end
    return (score, scoringParams, G)
end

function makeSingleChangeToGraph(G::DiGraph, data::DataFrame, oldScoringParams::ScoringParams, oldScore::Float64, searchComplete::Bool)
    n=oldScoringParams.n;
    parents=oldScoringParams.parents;
    currentNodeIndex = 1;
    while currentNodeIndex <= n
        resultsOfAddingParents= ResultsOfAddingParents([],[],ScoringParams[],DiGraph[]);
        for p=1:n
            newG=copy(G);
            add_edge!(newG, p, currentNodeIndex);
            # If this p is the current node or already a parent of the current node or it makes graph cyclic, skip it
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
    return ScoringParams(n,q,r,parents,M);
end

function updateScoringParams(df::DataFrame, oldParams::ScoringParams, i_node::Int64, i_parent::Int64)
    # Copying scoringParams into a new variable
    params=deepcopy(oldParams);
    n=params.n; # doesn't change
    r=params.r; # doesn't change
    q=params.q; # only q[i_node] changes
    parents=params.parents; # only parents[i_node] changes
    M=params.M; # only M[i_node] changes
    # changing parents
    push!(parents[i_node], i_parent);
    # changing q
    q[i_node]*=r[i_parent];
    # changing M
    M[i_node]=zeros(Int64, q[i_node], r[i_node]);
    parentIndices=parents[i_node];
    # aggregate data by itself and its parents
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

function calculateParentalInstantiationIndex(parentIndices, parentValues, r)
    r_parents=r[parentIndices]; # Number of instantiations of each parent
    A=fill(1,Tuple(r_parents)); # Array with dimensions corresponding to each parent
    linearIndices=LinearIndices(A);
    return linearIndices[parentValues...] # Using splat operator to index into linearIndices
end

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
    # data=CSV.File("large.csv") |> DataFrame;
    # n= length(data); # i = number of variables
    # G= DiGraph(n);
    # G1= DiGraph(n);
    # add_edge!(G1, 1, 2);
    # TODO Currently these take too long to run. Fix scoring function.
    # scForG=testScoringFunction(data,G,testCase=11,printingOn=true)
    # testScoringFunction(data,G1,testCase=8,printingOn=true)
end

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

function createAndRunTestsForK2Search()
    data=CSV.File("myownsmallexample.csv") |> DataFrame;
    println("Trying to find best graph for myownsmallexample.csv");
    (score, scoringParams)=performK2Search(data::DataFrame; maxNumberOfAttempts=10);
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
end

createAndRunTestsForScoringFunction()
createAndRunTestsForK2Search()
# gplot(G, nodelabel=1:n)

# if length(ARGS) != 2
#     error("usage: julia project1.jl <infile>.csv <outfile>.gph")
# end

# inputfilename = ARGS[1]
# outputfilename = ARGS[2]

inputfilename = ["small.csv", "medium.csv", "large.csv"]
outputfilename= ["small.gph", "medium.gph", "large.gph"]
for i=1:2
    compute(inputfilename[i], outputfilename[i])
end
