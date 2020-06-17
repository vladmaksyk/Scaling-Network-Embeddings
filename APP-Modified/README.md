
This is a C++ framework for variant **unweighted** network embedding techniques. We currently release the command line interface for following models:
- APP (**A**symmetric **P**roximity **P**reserving graph embedding)
  - [Scalable Graph Embedding for Asymmetric Proximity](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14696)

# Developed Environment
- g++ > 4.9 (In macOS, it needs OpenMP-enabled compilers. e.g. brew reinstall gcc6 --without-multilib)

# Compilation
```
$ cd APP-Modified
$ make
```


# Command Line Interface
Directly call the execution file to see the usage like:
```
./cli/app
```
then you will see the options description like:
```
Options Description:
        -train <string>
                Train the Network data
        -save <string>
                Save the representation data
        -dimensions <int>
                Dimension of vertex representation; default is 64
        -undirected <int>
                Whether the edge is undirected; default is 1
        -negative_samples <int>
                Number of negative examples; default is 5
        -window_size <int>
                Size of skip-gram window; default is 5
        -walk_times <int>
                Times of being staring vertex; default is 10
        -walk_steps <int>
                Step of random walk; default is 40
        -threads <int>
                Number of training threads; default is 1
        -alpha <float>
                Init learning rate; default is 0.025
Usage:
./app -train net.txt -save rep.txt -undirected 1 -dimensions 64 -walk_times 10 -walk_steps 40 -window_size 5 -negative_samples 5 -alpha 0.025 -threads 1
```



