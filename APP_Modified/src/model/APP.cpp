#include "APP.h"
#include <chrono> 
using namespace std::chrono;
APP::APP () {
}

APP::~APP () {
}

void APP::LoadEdgeList(string filename, bool undirect) {
    pnet.LoadEdgeList(filename, undirect);
}

void APP::SaveWeights(string model_name){

    cout << "Save Model:" << endl;
    ofstream model1(model_name+".src");
    ofstream model2(model_name+".tgt");
    if (model1 && model2)
    {
        model1 << pnet.MAX_vid << " " << dim << endl;
        model2 << pnet.MAX_vid << " " << dim << endl;

        for (long vid=0; vid!=pnet.MAX_vid; vid++)
        {
            model1 << pnet.vertex_hash.keys[vid];
            model2 << pnet.vertex_hash.keys[vid];

            for (int d=0; d<dim; ++d)
                {
                        model1 << " " << w_vertex[vid][d];
                        model2 << " " << w_context[vid][d];
                }
            model1 << endl;
            model2 << endl;
        }
        cout << "\tSave to <" << model_name << ">" << endl;
    }
    else
    {
        cout << "\tfail to open file" << endl;
    }
}

void APP::Init(int dim) {
    
    this->dim = dim;
    cout << "Model Setting:" << endl;
    cout << "\tdimension:\t\t" << dim << endl;
    
    w_vertex.resize(pnet.MAX_vid);
    w_context.resize(pnet.MAX_vid);

    for (long vid=0; vid<pnet.MAX_vid; ++vid)
    {
        w_vertex[vid].resize(dim);
        for (int d=0; d<dim;++d)
            w_vertex[vid][d] = (rand()/(double)RAND_MAX - 0.5) / dim;
    }

    for (long vid=0; vid<pnet.MAX_vid; ++vid)
    {
        w_context[vid].resize(dim);
        for (int d=0; d<dim;++d)
            w_context[vid][d] = (rand()/(double)RAND_MAX - 0.5) / dim;
    }
}


void APP::Train(int walk_times, int sample_times, double jump, int negative_samples, double alpha, int workers){
    
    omp_set_num_threads(workers);
    cout << "Model:" << endl;
    cout << "\t[APP]" << endl;
    cout << "Learning Parameters:" << endl;
    cout << "\twalk_times:\t\t" << walk_times << endl;
    cout << "\tsample_times:\t\t" << sample_times << endl;
    cout << "\tjumping factor:\t\t" << jump << endl;
    cout << "\tnegative_samples:\t" << negative_samples << endl;
    cout << "\talpha:\t\t\t" << alpha << endl;
    cout << "\tworkers:\t\t" << workers << endl;
    cout << "Start Training:" << endl;

    unsigned long long total = (unsigned long long)walk_times*pnet.MAX_vid;
    double alpha_min = alpha*0.0001;
    double _alpha = alpha;
    unsigned long long count = 0;
    double walk_time = 0.0;
    double train_time = 0.0;
         
    std::vector<long> random_keys(pnet.MAX_vid);
    for (long vid = 0; vid < pnet.MAX_vid; vid++) {
        random_keys[vid] = vid;
    }
    for (long vid = 0; vid < pnet.MAX_vid; vid++) {
        int rdx = vid + rand() % (pnet.MAX_vid - vid); // careful here!
        swap(random_keys[vid], random_keys[rdx]);
    }
    int totalCount = 0;
	
    #pragma omp parallel for
    for (long vid=0; vid<pnet.MAX_vid; ++vid)
    {
        auto start = high_resolution_clock::now();
        vector<long> walks = pnet.AmortizedRandomWalk(random_keys[vid], walk_times * sample_times, jump);
        totalCount += (walks.size() - 1);
		auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        walk_time = walk_time + (duration.count()/1000000.0);
        start = high_resolution_clock::now();
        pnet.UpdatePair(w_vertex, w_context, walks[0], walks.back(), dim, negative_samples, _alpha);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        train_time = train_time + (duration.count()/1000000.0);
        count++;
        if (count % MONITOR == 0)
        {
            _alpha = alpha* ( 1.0 - (double)(count)/total);
            if (_alpha < alpha_min) _alpha = alpha_min;
            printf("\tAlpha: %.6f\tProgress: %.3f %%%c", _alpha, (double)(count)/total * 100, 13);
            fflush(stdout);
        }
    }
    cout << "total sample size " << totalCount << endl;

    printf("\tAlpha: %.6f\tProgress: 100.00 %%\n", _alpha);
    cout << "walk_time: " << walk_time << " train_time: " << train_time << endl;
}


