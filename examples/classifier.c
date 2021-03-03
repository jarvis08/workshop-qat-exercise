#include "darknet.h"

#include <sys/time.h>
#include <assert.h>

float *get_regression_values(char **labels, int n)
{
    float *v = calloc(n, sizeof(float));
    int i;
    for(i = 0; i < n; ++i){
        char *p = strchr(labels[i], ' ');
        *p = 0;
        v[i] = atof(p+1);
    }
    return v;
}

void train_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    srand(time(0));
    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int count = 0;
    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        if(net->random && count++%40 == 0){
            printf("Resizing\n");
            int dim = (rand() % 11 + 4) * 32;
            //if (get_current_batch(net)+200 > net->max_batches) dim = 608;
            //int dim = (rand() % 4 + 16) * 32;
            printf("%d\n", dim);
            args.w = dim;
            args.h = dim;
            args.size = dim;
            args.min = net->min_ratio*dim;
            args.max = net->max_ratio*dim;
            printf("%d %d\n", args.min, args.max);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for(i = 0; i < ngpus; ++i){
                resize_network(nets[i], dim, dim);
            }
            net = nets[0];
        }
        time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
        }
        if(get_current_batch(net) >= 10000 && get_current_batch(net)%2000 == 0){
            char buff[256];
            sprintf(buff, "%s/%s_%d.backup",backup_directory,base,get_current_batch(net));
            save_weights(net, buff);
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void train_classifier_seq(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.threads = 32;
    net->cur_idx = -1;
    args.cur_idx = &net->cur_idx;

    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data_sequence(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_sequence(args);

        time = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        free_data(train);

        char buff_logfile[256];
        sprintf(buff_logfile, "log/%s.json", base);
        FILE *fp_log = fopen(buff_logfile, "a");
        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %f rate\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net));
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);
        fclose(fp_log);

        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
        }
    }

    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void train_classifier_with_random_and_sequential_dataset(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    char *dataset = basecfg(datacfg);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.threads = 32;
    net->cur_idx = -1;
    args.cur_idx = &net->cur_idx;
    args.data_map = calloc(N, sizeof(int));
    int k;
    for(k = 0; k < N; k ++) args.data_map[k] = k;
    random_shuffle(args.data_map, N);

    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_random_data_sequence(args);
    int epoch = (*net->seen)/N;

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    char buff_log[256];
    char buff[256];
    time_t current_time;
    char* c_time_string;
	current_time = time(NULL);
	c_time_string = ctime(&current_time);
	sprintf(buff_log, ">> Start training.. %s", c_time_string);
	fprintf(stderr, "%s", buff_log);
	fprintf(fp_log, buff_log);

    // Init training
	double time;
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_random_data_sequence(args);

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        free_data(train);

        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %lf rate, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);


        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            random_shuffle(args.data_map, N);

            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();

            if(0==strcmp(dataset, "imagenet")) {
                if(epoch >= 30  && (epoch % 5) == 0){
                    sprintf(buff, "%s/%s_e%d.backup", backup_directory, base, epoch);
                    save_weights(net, buff);
                    sprintf(buff_log, "Saved checkpoint to %s, and validate the model.", buff);
                    fprintf(stderr, "%s", buff_log);
                    fprintf(fp_log, buff_log);
                    validate_imagenet_classifier(datacfg, cfgfile, buff);
                }
            }
        }
    }
    free(args.data_map);

    sprintf(buff_log, ">>> Pre-training done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void train_classifier_by_transfer_learning(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_transfer_learning_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.threads = 32;
    net->cur_idx = -1;
    args.cur_idx = &net->cur_idx;
    args.data_map = calloc(N, sizeof(int));
    int k;
    for(k = 0; k < N; k ++) args.data_map[k] = k;
    random_shuffle(args.data_map, N);

    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_random_data_sequence(args);
    int epoch = (*net->seen)/N;

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    // Init training
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_random_data_sequence(args);

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        free_data(train);

        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %lf rate, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);


        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            random_shuffle(args.data_map, N);

            char buff_log[256];
            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();
        }
    }
    free(args.data_map);

    char buff_log[256];
    sprintf(buff_log, ">>> Pre-training done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void train_tiny_imagenet_without_few_layers(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_transfer_learning_network(cfgfile, weightfile, clear);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.threads = 32;
    net->cur_idx = -1;
    args.cur_idx = &net->cur_idx;
    args.data_map = calloc(N, sizeof(int));
    int k;
    for(k = 0; k < N; k ++) args.data_map[k] = k;
    random_shuffle(args.data_map, N);

    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_random_data_sequence(args);
    int epoch = (*net->seen)/N;

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    // Init training
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_random_data_sequence(args);

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_imagenet_pretrained_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        free_data(train);

        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %lf rate, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);


        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            random_shuffle(args.data_map, N);

            char buff_log[256];
            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();
        }
    }
    free(args.data_map);

    char buff_log[256];
    sprintf(buff_log, ">>> Pre-training done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    char buff[256];
    sprintf(buff, "%s/%s.weights", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void quantization_aware_training(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_quantization_simulated_network_int8(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network_quant(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %f rate\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net));

        fprintf(stderr, "%s", buff_log);

        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
        }
    }
    char buff[256];
    char buff_qw[256];
    char buff_qsz[256];
    sprintf(buff, "%s/%s.qat.int8.weights", backup_directory, base);
    sprintf(buff_qw, "%s/%s.qat.int8.qweights", backup_directory, base);
    sprintf(buff_qsz, "%s/%s.qat.int8.qparams", backup_directory, base);
    save_weights_int8(net, buff, buff_qw, buff_qsz);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void quantization_aware_training_of_tiny_imagenet_int4(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.threads = 32;
    net->cur_idx = -1;
    args.cur_idx = &net->cur_idx;
    args.data_map = calloc(N, sizeof(int));
    int k;
    for(k = 0; k < N; k ++) args.data_map[k] = k;
    random_shuffle(args.data_map, N);

    args.hierarchy = net->hierarchy;
    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    
    // load data sequentially
    load_thread = load_random_data_sequence(args);
    *net->seen = 0;
    int epoch = (*net->seen)/N;

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    // Init training
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    while(1){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_random_data_sequence(args);

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_quantization_simulated_network_of_tiny_imagenet_int4(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_quantization_simulated_network_int4(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %lf rate, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);

        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            random_shuffle(args.data_map, N);

            char buff_log[256];
            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();
            
            // QAT only 1 epoch
            if(epoch == net->qat_end_epoch) break;
        }
    }
    free(args.data_map);

    char buff_log[256];
    sprintf(buff_log, ">>> [INT4] Fine-tuning of tiny-imagenet with QAT done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    char buff[256];
    char buff_qw[256];
    char buff_qsz[256];
    sprintf(buff, "%s/%s.qat.int4.weights", backup_directory, base);
    sprintf(buff_qw, "%s/%s.qat.int4.qweights", backup_directory, base);
    sprintf(buff_qsz, "%s/%s.qat.int4.qparams", backup_directory, base);
    save_weights_int4(net, buff, buff_qw, buff_qsz);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void quantization_aware_training_of_tiny_imagenet_int8(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.threads = 32;
    net->cur_idx = -1;
    args.cur_idx = &net->cur_idx;
    args.data_map = calloc(N, sizeof(int));
    int k;
    for(k = 0; k < N; k ++) args.data_map[k] = k;
    random_shuffle(args.data_map, N);

    args.hierarchy = net->hierarchy;
    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    
    // load data sequentially
    load_thread = load_random_data_sequence(args);
    *net->seen = 0;
    int epoch = (*net->seen)/N;

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    // Init training
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    while(1){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_random_data_sequence(args);

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_quantization_simulated_network_of_tiny_imagenet_int8(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_quantization_simulated_network_int8(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %lf rate, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);

        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            random_shuffle(args.data_map, N);

            char buff_log[256];
            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();
            
            // QAT only 1 epoch
            if(epoch == net->qat_end_epoch) break;
        }
    }
    free(args.data_map);

    char buff_log[256];
    sprintf(buff_log, ">>> [INT8] Fine-tuning with QAT done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    char buff[256];
    char buff_qsz[256];
    char buff_qw[256];
    sprintf(buff, "%s/%s.qat.int8.weights", backup_directory, base);
    sprintf(buff_qsz, "%s/%s.qat.int8.qparams", backup_directory, base);
    sprintf(buff_qw, "%s/%s.qat.int8.qweights", backup_directory, base);
    save_weights_int8(net, buff, buff_qw, buff_qsz);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void quantization_aware_training_int4(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.threads = 32;
    net->cur_idx = -1;
    args.cur_idx = &net->cur_idx;
    args.data_map = calloc(N, sizeof(int));
    int k;
    for(k = 0; k < N; k ++) args.data_map[k] = k;
    random_shuffle(args.data_map, N);

    args.hierarchy = net->hierarchy;
    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    
    // load data sequentially
    load_thread = load_random_data_sequence(args);
    *net->seen = 0;
    int epoch = (*net->seen)/N;

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    // Init training
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    while(1){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_random_data_sequence(args);

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_quantization_simulated_network_int4(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_quantization_simulated_network_int4(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %lf rate, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);

        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            random_shuffle(args.data_map, N);

            char buff_log[256];
            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();
            
            // QAT only 1 epoch
            if(epoch == net->qat_end_epoch) break;
        }
    }
    free(args.data_map);

    char buff_log[256];
    sprintf(buff_log, ">>> [INT4] Fine-tuning with QAT done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    char buff[256];
    char buff_qw[256];
    char buff_qsz[256];
    sprintf(buff, "%s/%s.qat.int4.weights", backup_directory, base);
    sprintf(buff_qw, "%s/%s.qat.int4.qweights", backup_directory, base);
    sprintf(buff_qsz, "%s/%s.qat.int4.qparams", backup_directory, base);
    save_weights_int4(net, buff, buff_qw, buff_qsz);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void quantization_aware_training_int8(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.threads = 32;
    net->cur_idx = -1;
    args.cur_idx = &net->cur_idx;
    args.data_map = calloc(N, sizeof(int));
    int k;
    for(k = 0; k < N; k ++) args.data_map[k] = k;
    random_shuffle(args.data_map, N);

    args.hierarchy = net->hierarchy;
    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    
    // load data sequentially
    load_thread = load_random_data_sequence(args);
    *net->seen = 0;
    int epoch = (*net->seen)/N;

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    // Init training
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    while(1){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_random_data_sequence(args);

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_quantization_simulated_network_int8(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_quantization_simulated_network_int8(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %lf rate, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);

        free_data(train);
        if(get_current_batch(net) == 79) break;
        //if(*net->seen/N > epoch){
        //    epoch = *net->seen/N;
        //    random_shuffle(args.data_map, N);

        //    char buff_log[256];
        //    sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
        //    fprintf(stderr, "%s", buff_log);
        //    fprintf(fp_log, buff_log);
        //    epoch_start = what_time_is_it_now();
        //    
        //    // QAT only 1 epoch
        //    if(epoch == net->qat_end_epoch) break;
        //}
    }
    free(args.data_map);

    char buff_log[256];
    sprintf(buff_log, ">>> [INT8] Fine-tuning with QAT done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    char buff[256];
    char buff_qsz[256];
    char buff_qw[256];
    sprintf(buff, "%s/%s.qat.int8.weights", backup_directory, base);
    sprintf(buff_qsz, "%s/%s.qat.int8.qparams", backup_directory, base);
    sprintf(buff_qw, "%s/%s.qat.int8.qweights", backup_directory, base);
    save_weights_int8(net, buff, buff_qw, buff_qsz);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void quantization_aware_training_with_batch_of_mixed_clusters_int4(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    
    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    // Set arguments for thread
    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.hierarchy = net->hierarchy;
    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    args.threads = 1;
    net->cur_idx = -1;
    args.cur_idx = &net->cur_idx;

    args.data_map = calloc(N, sizeof(int));
    for(i = 0; i < N; i ++) args.data_map[i] = i;
    random_shuffle(args.data_map, N);

    // Load train dataset's  cluster info
    char buff_log[256];
    char *dataset = basecfg(datacfg);
    char buff_ctr[256];
    sprintf(buff_ctr, "./python/cluster/%s/train_%s.txt", dataset, net->ctr_method);

    sprintf(buff_log, "[Cluster Info] %s\n", buff_ctr);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);

    FILE* fp_cluster;
    fp_cluster = fopen(buff_ctr, "r");
    int* train_cluster = calloc(N, sizeof(int));
    for (i = 0; i < N; i++) {
        fscanf(fp_cluster, "%d\n", &train_cluster[i]);
    }
    fclose(fp_cluster);
    args.ctr_map = train_cluster;
    args.batch_info = calloc(net->ctr_k * 2 + 1, sizeof(int)); // [ctr, #data, ctr, #data, ...]

    // e.g., b_size=128 ctr_k=4 -> [2, 100, 0, 28, -1, *, *, *, *]
    // (+1) for the case when all the clusters' data are included to current batch (to give -1 value at the last)
    // need to check (-1) value during ema to know that already calculated the last cluster in the batch

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;

    *net->seen = 0;
    int epoch = (*net->seen)/N;
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    // Init training
    load_thread = load_random_data_sequence_with_ctr_info(args);
    while(1){
        pthread_join(load_thread, 0);
        train = buffer;
        // Use another array for batch_info for CPU/GPU parallelism
        for(i = 0; i < net->ctr_k * 2 + 1; i++) {
            net->ctr_info[i] = args.batch_info[i];
        }
        load_thread = load_random_data_sequence_with_ctr_info(args);
        batch_start = what_time_is_it_now();

        float loss = 0;
        loss = qat_network_with_batch_of_mixed_clusters_int4(net, train);
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        buff_log[256];
        sprintf(buff_log, "[Step: %ld, Epoch: %.3f] Loss: %f, Avg-loss: %f, LR: %lf, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);

        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            random_shuffle(args.data_map, N);

            buff_log[256];
            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();
            
            // QAT only 1 epoch
            if(epoch == net->qat_end_epoch) break;
        }
    }
    free(args.data_map);
    free(train_cluster);

    buff_log[256];
    sprintf(buff_log, ">>> QAT per cluster done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    save_cluster_qparams_int4(net, backup_directory, base, net->n);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void quantization_aware_training_with_single_cluster_info_int4(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];
    set_batch_network(net, 1);

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    
    // Load train dataset's  cluster info
    char *dataset = basecfg(datacfg);
    char buff_ctr[256];
    sprintf(buff_ctr, "./python/cluster/%s/train_%s.txt", dataset, net->ctr_method);
    FILE *fp_cluster;
    fp_cluster = fopen(buff_ctr, "r");
    int train_cluster[N];
    for (i = 0; i < N; i++) {
        fscanf(fp_cluster, "%d\n", &train_cluster[i]);
    }

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.hierarchy = net->hierarchy;
    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    args.threads = 1;
    net->cur_idx = -1;
    args.cur_idx = &net->cur_idx;
    args.data_map = calloc(N, sizeof(int));
    int k;
    for(k = 0; k < N; k ++) args.data_map[k] = k;
    random_shuffle(args.data_map, N);

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;

    // load data sequentially
    load_thread = load_random_data_sequence(args);

    fclose(fp_cluster);
    *net->seen = 0;
    int epoch = (*net->seen)/N;

    // Init training
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    int data_cluster;
    while(1){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_random_data_sequence(args);

        data_cluster = train_cluster[args.data_map[net->cur_idx]];

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        loss = qat_per_cluster_network_int4(net, train, data_cluster);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %lf rate, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);

        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
            random_shuffle(args.data_map, N);

            char buff_log[256];
            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();
            
            // QAT only 1 epoch
            if(epoch == net->qat_end_epoch) break;
        }
    }
    free(args.data_map);

    char buff_log[256];
    sprintf(buff_log, ">>> QAT per cluster done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    save_cluster_qparams_int4(net, backup_directory, base, net->n);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void quantization_aware_training_with_single_cluster_info_int8(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];
    set_batch_network(net, 1);

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.threads = 1;
    net->cur_idx = 0;
    args.cur_idx = &net->cur_idx;

    args.hierarchy = net->hierarchy;
    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    
    // Load train dataset's  cluster info
    char *dataset = basecfg(datacfg);
    char buff_ctr[256];
    sprintf(buff_ctr, "./python/cluster/%s/train_%s.txt", dataset, net->ctr_method);
    FILE *fp_cluster;
    fp_cluster = fopen(buff_ctr, "r");
    int train_cluster[N];
    for (i = 0; i < N; i++) {
        fscanf(fp_cluster, "%d\n", &train_cluster[i]);
    }

    fclose(fp_cluster);
    // load data sequentially
    load_thread = load_data_sequence(args);
    *net->seen = 0;
    int epoch = (*net->seen)/N;

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    // Init training
    int data_cluster;
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    while(1){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_sequence(args);

        int step = get_current_batch(net);
        data_cluster = train_cluster[step % N];
        printf("Cur.data's cluster = %d\n", data_cluster);

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = qat_per_cluster_network_int8(net, train, data_cluster);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = qat_per_cluster_network_int8(net, train, data_cluster);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %lf rate, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);

        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;

            char buff_log[256];
            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();
            
            // QAT only 1 epoch
            if(epoch == net->qat_end_epoch) break;
        }
    }
    free(args.data_map);

    char buff_log[256];
    sprintf(buff_log, ">>> QAT per cluster done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    save_cluster_qparams(net, backup_directory, base, net->n);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void quantization_aware_training_with_batch_cluster_info_int4(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i, j;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    
    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.hierarchy = net->hierarchy;
    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }


    // Load train dataset's  cluster info
    char *dataset = basecfg(datacfg);
    char buff_ctr[256];
    FILE *fp_cluster;

    net->cur_idx = 0;
    cluster ctrs[net->ctr_k];
    for( i = 0; i < net->ctr_k; i++ ) {
        sprintf(buff_ctr, "./python/cluster/%s/train_%s.c%d.txt", dataset, net->ctr_method, i);
        fp_cluster = fopen(buff_ctr, "r");
        fscanf(fp_cluster, "%d\n", &ctrs[i].size); // first line(int) of the file is the number of data
        ctrs[i].map = calloc(ctrs[i].size, sizeof(int));
        for (j = 0; j < ctrs[i].size; j++) {
            fscanf(fp_cluster, "%d\n", &ctrs[i].map[j]);
        }
        ctrs[i].idx = -1;
        ctrs[i].done = 0;
        fclose(fp_cluster);
        random_shuffle(ctrs[i].map, ctrs[i].size);
    }

    args.threads = 1;
    args.cur_idx = &net->cur_idx;
    args.clusters = &ctrs;
    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;

    // load data sequentially
    load_thread = load_cluster_data(args);

    *net->seen = 0;
    int epoch = 0;
    int cnt = 0;

    // Init training
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    while(1){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_cluster_data(args);

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        loss = qat_per_cluster_network_int4(net, train, net->cur_idx);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        char buff_log[256];
        sprintf(buff_log, "[Step: %ld, Epoch: %.3f, CTR: %d] Loss: %f, Avg-loss: %f, LR: %lf, %lf sec\n", get_current_batch(net), net->cur_idx, (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);

        free_data(train);

        cnt = 0;
        for( i = 0; i < net->ctr_k; i++) {
            if(!ctrs[i].done) cnt += 1;
        }

        if(!cnt) {
            ++epoch;
            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();
            
            if(epoch == net->qat_end_epoch) break;

            for( i = 0; i < net->ctr_k; i++) {
                ctrs[i].done = 0;
                random_shuffle(ctrs[i].map, ctrs[i].size);
            }
            cnt = net->ctr_k;
        }

        int on_going[cnt];
        j = 0;
        for( i = 0; i < net->ctr_k; i++) {
            if(!ctrs[i].done) {
                on_going[j++] = i;
            }
        }
        random_shuffle(&on_going, cnt);
        net->cur_idx = on_going[0]; // randomly chosen, next batch's cluster number
    }
    for( i = 0; i < net->ctr_k; i++) free(ctrs[i].map);

    char buff_log[256];
    sprintf(buff_log, ">>> QAT per cluster done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    save_cluster_qparams_int4(net, backup_directory, base, net->n);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void quantization_aware_training_with_batch_cluster_info_int8(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];
    set_batch_network(net, 1);

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.threads = 1;
    net->cur_idx = 0;
    args.cur_idx = &net->cur_idx;

    args.hierarchy = net->hierarchy;
    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    
    // Load train dataset's  cluster info
    char *dataset = basecfg(datacfg);
    char buff_ctr[256];
    sprintf(buff_ctr, "./python/cluster/%s/train_%s.txt", dataset, net->ctr_method);
    FILE *fp_cluster;
    fp_cluster = fopen(buff_ctr, "r");
    int train_cluster[N];
    for (i = 0; i < N; i++) {
        fscanf(fp_cluster, "%d\n", &train_cluster[i]);
    }

    fclose(fp_cluster);
    // load data sequentially
    load_thread = load_data_sequence(args);
    *net->seen = 0;
    int epoch = (*net->seen)/N;

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    // Init training
    int data_cluster;
    double train_start, epoch_start, batch_start;
    train_start = what_time_is_it_now();
    epoch_start = what_time_is_it_now();
    while(1){
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_sequence(args);

        int step = get_current_batch(net);
        data_cluster = train_cluster[step % N];
        printf("Cur.data's cluster = %d\n", data_cluster);

        batch_start = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = qat_per_cluster_network_int8(net, train, data_cluster);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = qat_per_cluster_network_int8(net, train, data_cluster);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        char buff_log[256];
        sprintf(buff_log, "%ld, %.3f: %f loss, %f avg, %lf rate, %lf sec\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now() - batch_start);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);

        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;

            char buff_log[256];
            sprintf(buff_log, ">>> %d epoch done, %lf sec\n", epoch, what_time_is_it_now() - epoch_start);
            fprintf(stderr, "%s", buff_log);
            fprintf(fp_log, buff_log);
            epoch_start = what_time_is_it_now();
            
            // QAT only 1 epoch
            if(epoch == net->qat_end_epoch) break;
        }
    }
    free(args.data_map);

    char buff_log[256];
    sprintf(buff_log, ">>> QAT per cluster done, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    save_cluster_qparams(net, backup_directory, base, net->n);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void minmax_qparams_through_one_epoch(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    // sequentially load data
    args.threads = 32;
    net->cur_idx = 0;
    args.cur_idx = &net->cur_idx;

    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;

    // load data sequentially
    load_thread = load_data_sequence(args);
    int epoch = 0;
    *net->seen = 0;

    // Open log file.
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");

    // Init training
    double train_start;
    train_start = what_time_is_it_now();

    printf("Setting quantization parameters..\n");
    while(1) {
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_sequence(args);
        printf("Current batch = %d\n", get_current_batch(net));

        min_max_through_dataset(net, train);
        free_data(train);

        if(get_current_batch(net) > N / net->batch) {
            break;
        }
    }
    char buff_log[256];
    sprintf(buff_log, ">>> Get qparams through 1 epoch, %lf seconds.\n", what_time_is_it_now() - train_start);
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);

    char buff_qsz[256];
    char buff_qw[256];
    sprintf(buff_qw, "%s/%s.qweights", backup_directory, base);
    sprintf(buff_qsz, "%s/%s.qparams", backup_directory, base);
    save_qparams(net, buff_qw, buff_qsz);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void train_qparams_per_cluster(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{

    FILE *fp_cluster;
    fp_cluster = fopen("./python/cluster/train_cluster.txt", "r");
    int train_cluster[50000];
    int i;
    for (i = 0; i < 50000; i++) {
        fscanf(fp_cluster, "%d\n", &train_cluster[i]);
    }
    fclose(fp_cluster);

    int cluster = 0;
    printf("Start setting qparams for Cluster %d\n", cluster);

    while(cluster < 10) {
        int i;
        char *base = basecfg(cfgfile);

        // Set name to load file
        sprintf(base, "%s_0", base);

        printf("%s\n", base);
        printf("%d\n", ngpus);
        network **nets = calloc(ngpus, sizeof(network*));

        srand(time(0));
        int seed = rand();
        for(i = 0; i < ngpus; ++i){
            srand(seed);
#ifdef GPU
            cuda_set_device(gpus[i]);
#endif
            nets[i] = load_network_qparams(cfgfile, weightfile, argv);
            nets[i]->learning_rate *= ngpus;
        }
        srand(time(0));
        network *net = nets[0];
        set_batch_network(net, 1);

        int imgs = net->batch * net->subdivisions * ngpus;

        list *options = read_data_cfg(datacfg);

        char *backup_directory = option_find_str(options, "backup", "/backup/");
        int tag = option_find_int_quiet(options, "tag", 0);
        char *label_list = option_find_str(options, "labels", "data/labels.list");
        char *train_list = option_find_str(options, "train", "data/train.list");
        char *tree = option_find_str(options, "tree", 0);
        if (tree) net->hierarchy = read_tree(tree);
        int classes = option_find_int(options, "classes", 2);

        char **labels = 0;
        if(!tag){
            labels = get_labels(label_list);
        }
        list *plist = get_paths(train_list);
        char **paths = (char **)list_to_array(plist);
        printf("%d\n", plist->size);
        int N = plist->size;

        load_args args = {0};
        args.w = net->w;
        args.h = net->h;

        // sequentially load data
        args.threads = 1;
        net->cur_idx = 0;
        args.cur_idx = &net->cur_idx;

        args.hierarchy = net->hierarchy;

        args.min = net->min_ratio*net->w;
        args.max = net->max_ratio*net->w;
        printf("%d %d\n", args.min, args.max);
        args.angle = net->angle;
        args.aspect = net->aspect;
        args.exposure = net->exposure;
        args.saturation = net->saturation;
        args.hue = net->hue;
        args.size = net->w;

        args.paths = paths;
        args.classes = classes;
        args.n = imgs;
        args.m = N;
        args.labels = labels;
        if (tag){
            args.type = TAG_DATA;
        } else {
            args.type = CLASSIFICATION_DATA;
        }

        data train;
        data buffer;
        pthread_t load_thread;
        args.d = &buffer;
        load_thread = load_data_sequence(args);

        *net->seen = 0;
        while(*net->seen < 50000) {
            pthread_join(load_thread, 0);
            train = buffer;
            load_thread = load_data_sequence(args);

            int step = get_current_batch(net);
            int data_cluster = train_cluster[step];
            printf("Current Global Cluster = %d\n", cluster);
            printf("Current Data Index = %d\n", step);
            printf("Current Data Cluster = %d\n", data_cluster);
            if(data_cluster == cluster) {
                ema_through_dataset(net, train);
            } else {
                *net->seen += 1;
            }
            free_data(train);
        }
        pthread_join(load_thread, 0);

        char buff_qsz[256];
        char buff_qw[256];
        sprintf(buff_qsz, "%s/%s.cluster_%d_qparams", backup_directory, base, cluster);
        sprintf(buff_qw, "%s/%s.cluster_%d_qweights", backup_directory, base, cluster);
        save_qparams(net, buff_qw, buff_qsz);


        free_network(net);
        if(labels) free_ptrs((void**)labels, classes);
        free_ptrs((void**)paths, plist->size);
        free_list(plist);
        free(base);
        cluster += 1;
    }
}

void batch_qparams(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    // sequentially load data
    args.threads = 32;
    net->cur_idx = 0;
    args.cur_idx = &net->cur_idx;

    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;

    // load data sequentially
    load_thread = load_data_sequence(args);

    int epoch = 0;
    *net->seen = 0;
    printf("Setting quantization parameters..\n");
    while(1) {
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_sequence(args);

        batch_min_max_through_dataset(net, train);

        free_data(train);
    }
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void input_qparams(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    // sequentially load data
    args.threads = 32;
    net->cur_idx = 0;
    args.cur_idx = &net->cur_idx;
    args.data_map = calloc(N, sizeof(int));
    int k;
    for(k = 0; k < N; k ++) args.data_map[k] = k;
    random_shuffle(args.data_map, N);

    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;

    // load data sequentially
    load_thread = load_random_data_sequence(args);

    *net->seen = 0;
    printf("Setting quantization parameters..\n");
    printf("Set %d sets of qparams..\n", net->n_qsz_set);
    while(*net->seen < net->n_qsz_set) {
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_random_data_sequence(args);

        printf("Current batch = %d\n", get_current_batch(net) + 1);
        get_qparams_of_inputs(net, train, base);
        free_data(train);
    }
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void train_qparams_staged(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_qparams(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    *net->seen = 0;
    int stage = 0;
    // To initialize the exit condition
    while(stage < 4) {
        time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        //printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        train_network_qparams_staged(net, train, stage);

        free_data(train);
        printf("stage: %d, cur_batch: %ld, epoch: %.3f, %lf seconds, %ld images\n", stage, get_current_batch(net), (float)(*net->seen)/N, what_time_is_it_now()-time, *net->seen);
        if(*net->seen/N > 0) {
            printf("Stage %d (layer %d) done.\n", stage, stage);
            set_quantization_stage(net, stage);
            stage += 1;
            *net->seen = 0;
        }
    }
    char buff_qsz[256];
    char buff_qw[256];
    sprintf(buff_qsz, "%s/%s.staged_qparams", backup_directory, base);
    sprintf(buff_qw, "%s/%s.staged_qweights", backup_directory, base);
    save_qparams(net, buff_qw, buff_qsz);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void minmax_qparams_without_transpose_info(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;
    char *base = basecfg(cfgfile);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        if(weightfile && weightfile[0] != 0) load_weights_without_xpose_info(nets[i], weightfile, 0, nets[i]->n);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    // To initialize the exit condition
    while(1) {
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        min_max_through_dataset(net, train);

        free_data(train);
        if(*net->seen/N > epoch){
            printf("epoch = %d\n", epoch);
            break;
        }
        //else printf("%d data seen of %d total..", *net->seen, N);
    }
    char buff_qsz[256];
    char buff_qw[256];
    sprintf(buff_qsz, "%s/%s.qparams", backup_directory, base);
    sprintf(buff_qw, "%s/%s.qweights", backup_directory, base);
    save_qparams(net, buff_qw, buff_qsz);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void txt_to_bin(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus)
{
    int i;
    char *base = basecfg(cfgfile);
    network **nets = calloc(ngpus, sizeof(network*));

    int seed = rand();
    for(i = 0; i < ngpus; ++i){
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_from_txt(cfgfile, weightfile, 0);
    }
    network *net = nets[0];

    list *options = read_data_cfg(datacfg);
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char buff[256];
    char *fname = basecfg(weightfile);
    sprintf(buff, "%s/%s.pretrained.weights", backup_directory, fname);
    save_converted_weights(net, buff);

    free_network(net);
    free(base);
}

void prune_weights(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;
    char *base = basecfg(cfgfile);
    network **nets = calloc(ngpus, sizeof(network*));

    for(i = 0; i < ngpus; ++i){
        nets[i] = load_network(cfgfile, weightfile, clear);
    }
    network *net = nets[0];

    list *options = read_data_cfg(datacfg);
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char buff[256];
    char pruning_buff[256];
    sprintf(buff, "%s/%s.pruned", backup_directory, base);
    sprintf(pruning_buff, "%s/%s.pparams", backup_directory, base);
    save_pruned_weights(net, buff, pruning_buff, net->n, argv);
    //load_weights_and_quantize(net, weightfile, 0, net->n);

    free_network(net);
    free(base);
}

void tune_weights(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;

    float avg_loss = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_pruned_network(cfgfile, weightfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    while(get_current_batch(net) < net->max_batches || net->max_batches == 0){
        time = what_time_is_it_now();

        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);
        time = what_time_is_it_now();

        float loss = 0;
#ifdef GPU
        if(ngpus == 1){
            loss = train_tuned_network(net, train);
        } else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network_quant(net, train);
#endif
        if(avg_loss == -1) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;
        printf("%ld, %.3f: %f, %f avg, %f rate, %lf seconds, %ld images\n", get_current_batch(net), (float)(*net->seen)/N, loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, *net->seen);
        free_data(train);
        if(*net->seen/N > epoch){
            epoch = *net->seen/N;
        }
    }
    char buff[256];
    sprintf(buff, "%s/%s.tuned", backup_directory, base);
    save_weights(net, buff);
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void tuned_qparams(char *datacfg, char *cfgfile, char *weightfile, char *pruningfile, int *gpus, int ngpus, int clear, char *argv)
{
    int i;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    printf("%d\n", ngpus);
    network **nets = calloc(ngpus, sizeof(network*));

    srand(time(0));
    int seed = rand();
    for(i = 0; i < ngpus; ++i){
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = load_network_pruned_qparams(cfgfile, weightfile, pruningfile, argv);
        nets[i]->learning_rate *= ngpus;
    }
    srand(time(0));
    network *net = nets[0];

    int imgs = net->batch * net->subdivisions * ngpus;

    list *options = read_data_cfg(datacfg);

    char *backup_directory = option_find_str(options, "backup", "/backup/");
    int tag = option_find_int_quiet(options, "tag", 0);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *train_list = option_find_str(options, "train", "data/train.list");
    char *tree = option_find_str(options, "tree", 0);
    if (tree) net->hierarchy = read_tree(tree);
    int classes = option_find_int(options, "classes", 2);

    char **labels = 0;
    if(!tag){
        labels = get_labels(label_list);
    }
    list *plist = get_paths(train_list);
    char **paths = (char **)list_to_array(plist);
    printf("%d\n", plist->size);
    int N = plist->size;
    double time;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.threads = 32;
    args.hierarchy = net->hierarchy;

    args.min = net->min_ratio*net->w;
    args.max = net->max_ratio*net->w;
    printf("%d %d\n", args.min, args.max);
    args.angle = net->angle;
    args.aspect = net->aspect;
    args.exposure = net->exposure;
    args.saturation = net->saturation;
    args.hue = net->hue;
    args.size = net->w;

    args.paths = paths;
    args.classes = classes;
    args.n = imgs;
    args.m = N;
    args.labels = labels;
    if (tag){
        args.type = TAG_DATA;
    } else {
        args.type = CLASSIFICATION_DATA;
    }

    data train;
    data buffer;
    pthread_t load_thread;
    args.d = &buffer;
    load_thread = load_data(args);

    int epoch = (*net->seen)/N;
    // To initialize the exit condition
    while(1) {
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data(args);

        min_max_through_dataset(net, train);

        free_data(train);
        if(*net->seen/N > epoch){
            printf("epoch = %d", epoch);
            break;
        }
        //else printf("%d data seen of %d total..", *net->seen, N);
    }
    char buff_qsz[256];
    char buff_qw[256];
    sprintf(buff_qsz, "%s/%s.tqparams", backup_directory, base);
    sprintf(buff_qw, "%s/%s.tqweights", backup_directory, base);
    save_tuned_qparams(net, buff_qw, buff_qsz, net->n); 
    pthread_join(load_thread, 0);

    free_network(net);
    if(labels) free_ptrs((void**)labels, classes);
    free_ptrs((void**)paths, plist->size);
    free_list(plist);
    free(base);
}

void validate_classifier_crop(char *datacfg, char *filename, char *weightfile)
{
    int i = 0;
    network *net = load_network(filename, weightfile, 0);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;
    float avg_acc = 0;
    float avg_topk = 0;
    int splits = m/1000;
    int num = (i+1)*m/splits - i*m/splits;

    data val, buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;

    args.paths = paths;
    args.classes = classes;
    args.n = num;
    args.m = 0;
    args.labels = labels;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(i = 1; i <= splits; ++i){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        num = (i+1)*m/splits - i*m/splits;
        char **part = paths+(i*m/splits);
        if(i != splits){
            args.paths = part;
            load_thread = load_data_in_thread(args);
        }
        printf("Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        float *acc = network_accuracies(net, val, topk);
        avg_acc += acc[0];
        avg_topk += acc[1];
        printf("%d: top 1: %f, top %d: %f, %lf seconds, %d images\n", i, avg_acc/i, topk, avg_topk/i, sec(clock()-time), val.X.rows);
        free_data(val);
    }
}

void validate_classifier_10(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        int w = net->w;
        int h = net->h;
        int shift = 32;
        image im = load_image_color(paths[i], w+shift, h+shift);
        image images[10];
        images[0] = crop_image(im, -shift, -shift, w, h);
        images[1] = crop_image(im, shift, -shift, w, h);
        images[2] = crop_image(im, 0, 0, w, h);
        images[3] = crop_image(im, -shift, shift, w, h);
        images[4] = crop_image(im, shift, shift, w, h);
        flip_image(im);
        images[5] = crop_image(im, -shift, -shift, w, h);
        images[6] = crop_image(im, shift, -shift, w, h);
        images[7] = crop_image(im, 0, 0, w, h);
        images[8] = crop_image(im, -shift, shift, w, h);
        images[9] = crop_image(im, shift, shift, w, h);
        float *pred = calloc(classes, sizeof(float));
        for(j = 0; j < 10; ++j){
            float *p = network_predict(net, images[j].data);
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1, 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            free_image(images[j]);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_classifier_full(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    int size = net->w;
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, size);
        resize_network(net, resized.w, resized.h);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, resized.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(resized);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}


void validate_classifier_single(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    // Log 9999th result
    char *base = basecfg(filename);
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");
    char buff_log[256];
    sprintf(buff_log, ">>> FP Validation with %s\n", weightfile);
    fprintf(fp_log, buff_log);

    double valid_start;
    valid_start = what_time_is_it_now();
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        sprintf(buff_log, "%d: top 1: %f, top %d: %f, %lf sec\n", i, avg_acc/(i+1), topk, avg_topk/(i+1), what_time_is_it_now() - valid_start);
        fprintf(stderr, "%s", buff_log);
    }
    fprintf(fp_log, buff_log);
    fclose(fp_log);
}

void validate_imagenet_classifier(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    // Log 9999th result
    char *base = basecfg(filename);
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");
    char buff_log[256];
    sprintf(buff_log, ">>> FP Validation with %s\n", weightfile);
    fprintf(fp_log, buff_log);

	//FILE * fp_labels = fopen("/data/ILSVRC2012_img_val/ILSVRC2012_validation_ground_truth.txt", "r");
	FILE * fp_labels = fopen("/home/ken/server-1-data/ILSVRC2012_img_val/ILSVRC2012_validation_ground_truth.txt", "r");
	int labels_list[m];
	for( i = 0; i < m; i++) {
    	fscanf(fp_labels, "%d\n", &labels_list[i]);
	}
	fclose(fp_labels);

    double valid_start;
    valid_start = what_time_is_it_now();
    char buff_path[256];
    for(i = 0; i < m; ++i){
        int class = -1;
        sprintf(buff_path, "%s", paths[i]);
        char *splitted = strtok(buff_path, "/");
        while( splitted != NULL ) {
            if( strstr(splitted, "ILSVRC2012_val") != NULL ) {
				char *tmp = replaceString(splitted, "ILSVRC2012_val_", "");
				tmp = replaceString(tmp, ".JPEG", "");
				class = labels_list[atoi(tmp) - 1] - 1;
				break;
            }
            splitted = strtok(NULL, "/");
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        sprintf(buff_log, "%d: top 1: %f, top %d: %f, %lf sec\n", i, avg_acc/(i+1), topk, avg_topk/(i+1), what_time_is_it_now() - valid_start);
        fprintf(stderr, "%s", buff_log);
    }
    fprintf(fp_log, buff_log);
    fclose(fp_log);
}

void validate_imagenet_pretrained_classifier(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = parse_network_cfg(filename);
    if(weightfile && weightfile[0] != 0) load_weights_without_xpose_info(net, weightfile, 0, net->n);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    // Log 9999th result
    char *base = basecfg(filename);
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");
    char buff_log[256];
    sprintf(buff_log, ">>> FP Validation with %s\n", weightfile);
    fprintf(fp_log, buff_log);

	FILE * fp_labels = fopen("/home/ken/server-1-data/ILSVRC2012_img_val/ILSVRC2012_validation_ground_truth.txt", "r");
	int labels_list[m];
	for( i = 0; i < m; i++) {
        if( i != m - 1) fscanf(fp_labels, "%d\n", &labels_list[i]);
        else fscanf(fp_labels, "%d", &labels_list[i]);
    }
	fclose(fp_labels);

    double valid_start;
    valid_start = what_time_is_it_now();
    char buff_path[256];
    for(i = 0; i < m; ++i){
        int class = -1;
        sprintf(buff_path, "%s", paths[i]);
        char *splitted = strtok(buff_path, "/");
        while( splitted != NULL ) {
            if( strstr(splitted, "ILSVRC2012_val") != NULL ) {
				char *tmp = replaceString(splitted, "ILSVRC2012_val_", "");
				tmp = replaceString(tmp, ".JPEG", "");
				class = labels_list[atoi(tmp) - 1];
				break;
            }
            splitted = strtok(NULL, "/");
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        sprintf(buff_log, "%d: top 1: %f, top %d: %f, %lf sec\n", i, avg_acc/(i+1), topk, avg_topk/(i+1), what_time_is_it_now() - valid_start);
        fprintf(stderr, "%s", buff_log);
    }
    fprintf(fp_log, buff_log);
    fclose(fp_log);
}

void validate_classifier_without_transpose_info(char *datacfg, char *filename, char *weightfile)
{
    int i, j;
    network *net = parse_network_cfg(filename);
    if(weightfile && weightfile[0] != 0) load_weights_without_xpose_info(net, weightfile, 0, net->n);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        //grayscale_image_3c(crop);
        //show_image(im, "orig");
        //show_image(crop, "cropped");
        //cvWaitKey(0);
        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_classifier_single_quant(char *datacfg, char *networkcfg, char *weightfile, char *qweightfile, char *qszfile, char *argv)
{
    int i, j;
    network *net = load_quantized_network(networkcfg, weightfile, qweightfile, qszfile, 0, argv);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict_int8(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void validate_quantized_network_with_single_cluster_info_int4(char *datacfg, char *networkcfg, char *argv)
{
    int i, j;
    network *net = load_network_qparams(networkcfg, 0, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    // Load input qsz sets
    char *dataset = basecfg(datacfg);
    char buff_ctr[256];
    sprintf(buff_ctr, "./python/cluster/%s/test_%s.txt", dataset, net->ctr_method);
    FILE *fp_cluster;
    fp_cluster = fopen(buff_ctr, "r");
    int test_cluster[m];
    int *ctr_el_cnt = calloc(net->ctr_k, sizeof(int));
    for (i = 0; i < m; i++) {
        fscanf(fp_cluster, "%d\n", &test_cluster[i]);
        ctr_el_cnt[test_cluster[i]] += 1;
    }
    fclose(fp_cluster);

    // Set name to load file
    char *base = basecfg(networkcfg);

    // Log 9999th result
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");
    char buff_log[256];
    sprintf(buff_log, ">>> [Cluster INT4] Validation with %s\n", base);
    fprintf(fp_log, buff_log);

    double valid_start;
    valid_start = what_time_is_it_now();
    int prev_cluster = -1;
    int data_cluster = -1;
    int label_right[10] = {0}; // Must be modified as # of labels
    int *ctr_right = calloc(net->ctr_k, sizeof(int));
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        data_cluster = test_cluster[i];
        if(data_cluster != prev_cluster) {
            printf("data cluster = %d\n", data_cluster);
            char buff_qweight[256];
            char buff_qsz[256];
            sprintf(buff_qweight, "backup/%s.%s.k%d.c%d.int4.qweights", base, net->ctr_method, net->ctr_k, data_cluster);
            sprintf(buff_qsz, "backup/%s.%s.k%d.c%d.int4.qparams", base, net->ctr_method, net->ctr_k, data_cluster);
            //sprintf(buff_qweight, "backup/%s.qat.c%d.int4.qweights", base, data_cluster);
            //sprintf(buff_qsz, "backup/%s.qat.c%d.int4.qparams", base, data_cluster);
            load_quantized_weights_only_int4(net, buff_qweight, 0, net->n);
            load_qparams_only_int4(net, buff_qsz, 0, net->n); 
            prev_cluster = data_cluster;
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict_int4(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) {
            avg_acc += 1;
            label_right[indexes[0]] += 1;
            ctr_right[test_cluster[i]] += 1;
        }
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        sprintf(buff_log, "%d: top 1: %f, top %d: %f, %lf sec\n", i, avg_acc/(i+1), topk, avg_topk/(i+1), what_time_is_it_now() - valid_start);
        fprintf(stderr, "%s", buff_log);
    }
    fprintf(fp_log, buff_log);

    float *avg_ctr_right = calloc(net->ctr_k, sizeof(float));
    for (i = 0; i < net->ctr_k; i ++) {
        avg_ctr_right[i] = ctr_right[i];
        avg_ctr_right[i] /= ctr_el_cnt[i];
        avg_ctr_right[i] *= 100;
    }
    for (i = 0; i < net->ctr_k; i ++) {
        sprintf(buff_log, "#right cluster-%d: {%d}/{%d}, %f%%\n", i, ctr_right[i], ctr_el_cnt[i], avg_ctr_right[i]);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);
    }
    for (i = 0; i < 10; i ++) {
        sprintf(buff_log, "#right label %d: %d\n", i, label_right[i]);
        fprintf(stderr, "%s", buff_log);
        fprintf(fp_log, buff_log);
    }
    sprintf(buff_log, "\n");
    fprintf(stderr, "%s", buff_log);
    fprintf(fp_log, buff_log);
    fclose(fp_log);
    free(avg_ctr_right);
    free(ctr_el_cnt);
    free(ctr_right);
}

void validate_quantized_network_with_single_cluster_info_int8(char *datacfg, char *networkcfg, char *argv)
{
    int i, j;
    network *net = load_network_qparams(networkcfg, 0, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);
    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    // Load input qsz sets
    char *dataset = basecfg(datacfg);
    char buff_ctr[256];
    sprintf(buff_ctr, "./python/cluster/%s/test_%s.txt", dataset, net->ctr_method);
    FILE *fp_cluster;
    fp_cluster = fopen(buff_ctr, "r");
    int test_cluster[m];
    for (i = 0; i < m; i++) {
        fscanf(fp_cluster, "%d\n", &test_cluster[i]);
    }
    fclose(fp_cluster);

    // Set name to load file
    char *base = basecfg(networkcfg);
    //sprintf(base, "%s_0", base);

    // Log 9999th result
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");
    char buff_log[256];
    sprintf(buff_log, ">>> [Cluster INT8] Validation with %s\n", base);
    fprintf(fp_log, buff_log);

    double valid_start;
    valid_start = what_time_is_it_now();
    int prev_cluster = -1;
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        int data_cluster = test_cluster[i];
        if(data_cluster != prev_cluster) {
            printf("data cluster = %d\n", data_cluster);
            char buff_qweight[256];
            char buff_qsz[256];
            //sprintf(buff_qweight, "backup/%s.cluster_%d_qweights", base, data_cluster);
            //sprintf(buff_qsz, "backup/%s.cluster_%d_qparams", base, data_cluster);
            sprintf(buff_qweight, "backup/%s.qat.c%d.int8.qweights", base, data_cluster);
            sprintf(buff_qsz, "backup/%s.qat.c%d.int8.qparams", base, data_cluster);
            load_quantized_weights_only(net, buff_qweight, 0, net->n);
            load_qparams_only(net, buff_qsz, 0, net->n); 
#ifdef GPU
#ifndef CUDNN
            make_fake_integer_weights(net);
#endif
#endif
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        sprintf(buff_log, "%d: top 1: %f, top %d: %f, %lf sec\n", i, avg_acc/(i+1), topk, avg_topk/(i+1), what_time_is_it_now() - valid_start);
        fprintf(stderr, "%s", buff_log);
    }
    fprintf(fp_log, buff_log);
    fclose(fp_log);
}

void validate_via_qsz_selection(char *datacfg, char *networkcfg, char *weightfile, char *qweightfile, char *qszfile, char *argv)
{
    int i, j;
    network *net = load_quantized_network(networkcfg, weightfile, qweightfile, qszfile, 0, argv);

    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    // Load input qsz sets
    char *base = basecfg(networkcfg);
    char buff_qsz[256];
    char buff_ranges[256];
    sprintf(buff_qsz, "backup/%s.input_qparams", base);
    sprintf(buff_ranges, "backup/%s.input_ranges", base);
    load_input_qparams(net, buff_qsz, buff_ranges);

    // Log 9999th result
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");
    char buff_log[256];
    sprintf(buff_log, ">>> [QSZ Selection] INT Validation with %s\n", weightfile);
    fprintf(fp_log, buff_log);

    double valid_start;
    valid_start = what_time_is_it_now();
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict_with_closest_input(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        //printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
        sprintf(buff_log, "%d: top 1: %f, top %d: %f, %lf sec\n", i, avg_acc/(i+1), topk, avg_topk/(i+1), what_time_is_it_now() - valid_start);
        fprintf(stderr, "%s", buff_log);
    }
    fprintf(fp_log, buff_log);
    fclose(fp_log);
}

void validate_quantized_network_int4(char *datacfg, char *networkcfg, char *qweightfile, char *qszfile, char *argv)
{
    int i, j;
    network *net = load_quantized_network_int4(networkcfg, 0, qweightfile, qszfile, 0, argv);

    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    // Log 9999th result
    char *base = basecfg(networkcfg);
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");
    char buff_log[256];
    sprintf(buff_log, ">>> [INT4] Validation with %s, %s\n", qweightfile, qszfile);
    fprintf(fp_log, buff_log);

    double valid_start;
    valid_start = what_time_is_it_now();
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict_int4(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        sprintf(buff_log, "%d: top 1: %f, top %d: %f, %lf sec\n", i, avg_acc/(i+1), topk, avg_topk/(i+1), what_time_is_it_now() - valid_start);
        fprintf(stderr, "%s", buff_log);
    }
    fprintf(fp_log, buff_log);
    fclose(fp_log);
}

void validate_quantized_network_int8(char *datacfg, char *networkcfg, char *qweightfile, char *qszfile, char *argv)
{
    int i, j;
    network *net = load_quantized_network(networkcfg, 0, qweightfile, qszfile, 0, argv);

#ifdef GPU
    make_fake_integer_weights(net);
//#ifndef CUDNN
//    // fake int gemm with fp gemm_gpu
//    make_fake_integer_weights(net);
//#endif
#endif

    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    // Log 9999th result
    char *base = basecfg(networkcfg);
    char buff_logfile[256];
    sprintf(buff_logfile, "log/%s.json", base);
    FILE *fp_log = fopen(buff_logfile, "a");
    char buff_log[256];
    sprintf(buff_log, ">>> [INT8] Validation with %s, %s\n", qweightfile, qszfile);
    fprintf(fp_log, buff_log);

    double valid_start;
    valid_start = what_time_is_it_now();
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict_int8(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        //printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
        sprintf(buff_log, "%d: top 1: %f, top %d: %f, %lf sec\n", i, avg_acc/(i+1), topk, avg_topk/(i+1), what_time_is_it_now() - valid_start);
        fprintf(stderr, "%s", buff_log);
    }
    fprintf(fp_log, buff_log);
    fclose(fp_log);
}

void validate_pruned_classifier_int8(char *datacfg, char *networkcfg, char *qweightfile, char *qszfile, char *pruningfile, char *argv)
{
    int i, j;
    network *net = load_quantized_network_pruned(networkcfg, qweightfile, qszfile, pruningfile, 0, argv);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict_int8(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void compare_fp_and_int(char *datacfg, char *networkcfg, char *weightfile, char *qweightfile, char *qszfile, char *argv)
{
    int i, j;
    network *net = load_quantized_network(networkcfg, weightfile, qweightfile, qszfile, 0, argv);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));
    
    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict_comparison(net, crop.data);
        //for(j=0;j<10;j++) printf("pred = %f\n", pred[j]);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void save_into_csv(char *datacfg, char *networkcfg, char *weightfile, char *qweightfile, char *qszfile, char *is_csv, char *argv)
{
    int i, j;
    network *net = load_quantized_network(networkcfg, weightfile, qweightfile, qszfile, 0, argv);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *leaf_list = option_find_str(options, "leaves", 0);
    if(leaf_list) change_leaves(net->hierarchy, leaf_list);
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));
    
    // Write outputs into CSV file
    char *base = basecfg(networkcfg);
    FILE *output_fp = NULL;
    FILE *weight_fp = NULL;

    if(is_csv) {
        char output_buff[256];
        sprintf(output_buff, "%s/%s-output.csv", base, base);
        printf("CSV save mode >> %s\n", output_buff);
        output_fp = fopen(output_buff, "w");
        char weight_buff[256];
        //sprintf(weight_buff, "%s/%s-weight.csv", base, base);
        sprintf(weight_buff, "%s/clamping-test-weight-default.csv", base);
        printf("CSV save mode >> %s\n", weight_buff);
        weight_fp = fopen(weight_buff, "w");
    }

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        image im = load_image_color(paths[i], 0, 0);
        image crop = center_crop_image(im, net->w, net->h);
        float *pred = network_predict_csv(net, crop.data, output_fp, weight_fp);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 1, 1);

        free_image(im);
        free_image(crop);
        top_k(pred, classes, topk, indexes);

        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%s, %d, %f, %f, \n", paths[i], class, pred[0], pred[1]);
        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
    // Close CSV file
    if(is_csv) {
        fclose(output_fp);
        fclose(weight_fp);
    }
}

void validate_classifier_multi(char *datacfg, char *cfg, char *weights)
{
    int i, j;
    network *net = load_network(cfg, weights, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "labels", "data/labels.list");
    char *valid_list = option_find_str(options, "valid", "data/train.list");
    int classes = option_find_int(options, "classes", 2);
    int topk = option_find_int(options, "top", 1);

    char **labels = get_labels(label_list);
    list *plist = get_paths(valid_list);
    //int scales[] = {224, 288, 320, 352, 384};
    int scales[] = {224, 256, 288, 320};
    int nscales = sizeof(scales)/sizeof(scales[0]);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    float avg_acc = 0;
    float avg_topk = 0;
    int *indexes = calloc(topk, sizeof(int));

    for(i = 0; i < m; ++i){
        int class = -1;
        char *path = paths[i];
        for(j = 0; j < classes; ++j){
            if(strstr(path, labels[j])){
                class = j;
                break;
            }
        }
        float *pred = calloc(classes, sizeof(float));
        image im = load_image_color(paths[i], 0, 0);
        for(j = 0; j < nscales; ++j){
            image r = resize_max(im, scales[j]);
            resize_network(net, r.w, r.h);
            float *p = network_predict(net, r.data);
            if(net->hierarchy) hierarchy_predictions(p, net->outputs, net->hierarchy, 1 , 1);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            flip_image(r);
            p = network_predict(net, r.data);
            axpy_cpu(classes, 1, p, 1, pred, 1);
            if(r.data != im.data) free_image(r);
        }
        free_image(im);
        top_k(pred, classes, topk, indexes);
        free(pred);
        if(indexes[0] == class) avg_acc += 1;
        for(j = 0; j < topk; ++j){
            if(indexes[j] == class) avg_topk += 1;
        }

        printf("%d: top 1: %f, top %d: %f\n", i, avg_acc/(i+1), topk, avg_topk/(i+1));
    }
}

void try_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int layer_num)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image orig = load_image_color(input, 0, 0);
        image r = resize_min(orig, 256);
        image im = crop_image(r, (r.w - 224 - 1)/2 + 1, (r.h - 224 - 1)/2 + 1, 224, 224);
        float mean[] = {0.48263312050943, 0.45230225481413, 0.40099074308742};
        float std[] = {0.22590347483426, 0.22120921437787, 0.22103996251583};
        float var[3];
        var[0] = std[0]*std[0];
        var[1] = std[1]*std[1];
        var[2] = std[2]*std[2];

        normalize_cpu(im.data, mean, var, 1, 3, im.w*im.h);

        float *X = im.data;
        time=clock();
        float *predictions = network_predict(net, X);

        layer l = net->layers[layer_num];
        for(i = 0; i < l.c; ++i){
            if(l.rolling_mean) printf("%f %f %f\n", l.rolling_mean[i], l.rolling_variance[i], l.scales[i]);
        }
#ifdef GPU
        cuda_pull_array(l.output_gpu, l.output, l.outputs);
#endif
        for(i = 0; i < l.outputs; ++i){
            printf("%f\n", l.output[i]);
        }
        /*

           printf("\n\nWeights\n");
           for(i = 0; i < l.n*l.size*l.size*l.c; ++i){
           printf("%f\n", l.filters[i]);
           }

           printf("\n\nBiases\n");
           for(i = 0; i < l.n; ++i){
           printf("%f\n", l.biases[i]);
           }
         */

        top_predictions(net, top, indexes);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%s: %f\n", names[index], predictions[index]);
        }
        free_image(im);
        if (filename) break;
    }
}

void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top)
{
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top == 0) top = option_find_int(options, "top", 1);

    int i = 0;
    char **names = get_labels(name_list);
    clock_t time;
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        }else{
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input, 0, 0);
        image r = letterbox_image(im, net->w, net->h);
        //image r = resize_min(im, 320);
        //printf("%d %d\n", r.w, r.h);
        //resize_network(net, r.w, r.h);
        //printf("%d %d\n", r.w, r.h);

        float *X = r.data;
        time=clock();
        float *predictions = network_predict(net, X);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);
        fprintf(stderr, "%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        for(i = 0; i < top; ++i){
            int index = indexes[i];
            //if(net->hierarchy) printf("%d, %s: %f, parent: %s \n",index, names[index], predictions[index], (net->hierarchy->parent[index] >= 0) ? names[net->hierarchy->parent[index]] : "Root");
            //else printf("%s: %f\n",names[index], predictions[index]);
            printf("%5.2f%%: %s\n", predictions[index]*100, names[index]);
        }
        if(r.data != im.data) free_image(r);
        free_image(im);
        if (filename) break;
    }
}


void label_classifier(char *datacfg, char *filename, char *weightfile)
{
    int i;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *label_list = option_find_str(options, "names", "data/labels.list");
    char *test_list = option_find_str(options, "test", "data/train.list");
    int classes = option_find_int(options, "classes", 2);

    char **labels = get_labels(label_list);
    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);
        float *pred = network_predict(net, crop.data);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);
        int ind = max_index(pred, classes);

        printf("%s\n", labels[ind]);
    }
}

void csv_classifier(char *datacfg, char *cfgfile, char *weightfile)
{
    int i,j;
    network *net = load_network(cfgfile, weightfile, 0);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *test_list = option_find_str(options, "test", "data/test.list");
    int top = option_find_int(options, "top", 1);

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);
    int *indexes = calloc(top, sizeof(int));

    for(i = 0; i < m; ++i){
        double time = what_time_is_it_now();
        char *path = paths[i];
        image im = load_image_color(path, 0, 0);
        image r = letterbox_image(im, net->w, net->h);
        float *predictions = network_predict(net, r.data);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_k(predictions, net->outputs, top, indexes);

        printf("%s", path);
        for(j = 0; j < top; ++j){
            printf("\t%d", indexes[j]);
        }
        printf("\n");

        free_image(im);
        free_image(r);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", what_time_is_it_now() - time, i+1, m);
    }
}

void test_classifier(char *datacfg, char *cfgfile, char *weightfile, int target_layer)
{
    int curr = 0;
    network *net = load_network(cfgfile, weightfile, 0);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    char *test_list = option_find_str(options, "test", "data/test.list");
    int classes = option_find_int(options, "classes", 2);

    list *plist = get_paths(test_list);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    clock_t time;

    data val, buffer;

    load_args args = {0};
    args.w = net->w;
    args.h = net->h;
    args.paths = paths;
    args.classes = classes;
    args.n = net->batch;
    args.m = 0;
    args.labels = 0;
    args.d = &buffer;
    args.type = OLD_CLASSIFICATION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    for(curr = net->batch; curr < m; curr += net->batch){
        time=clock();

        pthread_join(load_thread, 0);
        val = buffer;

        if(curr < m){
            args.paths = paths + curr;
            if (curr + net->batch > m) args.n = m - curr;
            load_thread = load_data_in_thread(args);
        }
        fprintf(stderr, "Loaded: %d images in %lf seconds\n", val.X.rows, sec(clock()-time));

        time=clock();
        matrix pred = network_predict_data(net, val);

        int i, j;
        if (target_layer >= 0){
            //layer l = net->layers[target_layer];
        }

        for(i = 0; i < pred.rows; ++i){
            printf("%s", paths[curr-net->batch+i]);
            for(j = 0; j < pred.cols; ++j){
                printf("\t%g", pred.vals[i][j]);
            }
            printf("\n");
        }

        free_matrix(pred);

        fprintf(stderr, "%lf seconds, %d images, %d total\n", sec(clock()-time), val.X.rows, curr);
        free_data(val);
    }
}

void file_output_classifier(char *datacfg, char *filename, char *weightfile, char *listfile)
{
    int i,j;
    network *net = load_network(filename, weightfile, 0);
    set_batch_network(net, 1);
    srand(time(0));

    list *options = read_data_cfg(datacfg);

    //char *label_list = option_find_str(options, "names", "data/labels.list");
    int classes = option_find_int(options, "classes", 2);

    list *plist = get_paths(listfile);

    char **paths = (char **)list_to_array(plist);
    int m = plist->size;
    free_list(plist);

    for(i = 0; i < m; ++i){
        image im = load_image_color(paths[i], 0, 0);
        image resized = resize_min(im, net->w);
        image crop = crop_image(resized, (resized.w - net->w)/2, (resized.h - net->h)/2, net->w, net->h);

        float *pred = network_predict(net, crop.data);
        if(net->hierarchy) hierarchy_predictions(pred, net->outputs, net->hierarchy, 0, 1);

        if(resized.data != im.data) free_image(resized);
        free_image(im);
        free_image(crop);

        printf("%s", paths[i]);
        for(j = 0; j < classes; ++j){
            printf("\t%g", pred[j]);
        }
        printf("\n");
    }
}


void threat_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    float threat = 0;
    float roll = .2;

    printf("Classifier Demo\n");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    void * cap = open_video_stream(filename, cam_index, 0,0,0);

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    //cvNamedWindow("Threat", CV_WINDOW_NORMAL); 
    //cvResizeWindow("Threat", 512, 512);
    float fps = 0;
    int i;

    int count = 0;

    while(1){
        ++count;
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        if(!in.data) break;
        image in_s = resize_image(in, net->w, net->h);

        image out = in;
        int x1 = out.w / 20;
        int y1 = out.h / 20;
        int x2 = 2*x1;
        int y2 = out.h - out.h/20;

        int border = .01*out.h;
        int h = y2 - y1 - 2*border;
        int w = x2 - x1 - 2*border;

        float *predictions = network_predict(net, in_s.data);
        float curr_threat = 0;
        if(1){
            curr_threat = predictions[0] * 0 + 
                predictions[1] * .6 + 
                predictions[2];
        } else {
            curr_threat = predictions[218] +
                predictions[539] + 
                predictions[540] + 
                predictions[368] + 
                predictions[369] + 
                predictions[370];
        }
        threat = roll * curr_threat + (1-roll) * threat;

        draw_box_width(out, x2 + border, y1 + .02*h, x2 + .5 * w, y1 + .02*h + border, border, 0,0,0);
        if(threat > .97) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .02*h - 2*border, 
                    x2 + .5 * w + 6*border, 
                    y1 + .02*h + 3*border, 3*border, 1,0,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .02*h - 2*border, 
                x2 + .5 * w + 6*border, 
                y1 + .02*h + 3*border, .5*border, 0,0,0);
        draw_box_width(out, x2 + border, y1 + .42*h, x2 + .5 * w, y1 + .42*h + border, border, 0,0,0);
        if(threat > .57) {
            draw_box_width(out,  x2 + .5 * w + border,
                    y1 + .42*h - 2*border, 
                    x2 + .5 * w + 6*border, 
                    y1 + .42*h + 3*border, 3*border, 1,1,0);
        }
        draw_box_width(out,  x2 + .5 * w + border,
                y1 + .42*h - 2*border, 
                x2 + .5 * w + 6*border, 
                y1 + .42*h + 3*border, .5*border, 0,0,0);

        draw_box_width(out, x1, y1, x2, y2, border, 0,0,0);
        for(i = 0; i < threat * h ; ++i){
            float ratio = (float) i / h;
            float r = (ratio < .5) ? (2*(ratio)) : 1;
            float g = (ratio < .5) ? 1 : 1 - 2*(ratio - .5);
            draw_box_width(out, x1 + border, y2 - border - i, x2 - border, y2 - border - i, 1, r, g, 0);
        }
        top_predictions(net, top, indexes);
        char buff[256];
        sprintf(buff, "/home/pjreddie/tmp/threat_%06d", count);
        //save_image(out, buff);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        for(i = 0; i < top; ++i){
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);
        }

        if(1){
            show_image(out, "Threat", 10);
        }
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}


void gun_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    int bad_cats[] = {218, 539, 540, 1213, 1501, 1742, 1911, 2415, 4348, 19223, 368, 369, 370, 1133, 1200, 1306, 2122, 2301, 2537, 2823, 3179, 3596, 3639, 4489, 5107, 5140, 5289, 6240, 6631, 6762, 7048, 7171, 7969, 7984, 7989, 8824, 8927, 9915, 10270, 10448, 13401, 15205, 18358, 18894, 18895, 19249, 19697};

    printf("Classifier Demo\n");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);
    void * cap = open_video_stream(filename, cam_index, 0,0,0);

    int top = option_find_int(options, "top", 1);

    char *name_list = option_find_str(options, "names", 0);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        image in_s = resize_image(in, net->w, net->h);

        float *predictions = network_predict(net, in_s.data);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");

        int threat = 0;
        for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i){
            int index = bad_cats[i];
            if(predictions[index] > .01){
                printf("Threat Detected!\n");
                threat = 1;
                break;
            }
        }
        if(!threat) printf("Scanning...\n");
        for(i = 0; i < sizeof(bad_cats)/sizeof(bad_cats[0]); ++i){
            int index = bad_cats[i];
            if(predictions[index] > .01){
                printf("%s\n", names[index]);
            }
        }

        show_image(in, "Threat Detection", 10);
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}

void demo_classifier(char *datacfg, char *cfgfile, char *weightfile, int cam_index, const char *filename)
{
#ifdef OPENCV
    char *base = basecfg(cfgfile);
    image **alphabet = load_alphabet();
    printf("Classifier Demo\n");
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    list *options = read_data_cfg(datacfg);

    srand(2222222);

    int w = 1280;
    int h = 720;
    void * cap = open_video_stream(filename, cam_index, w, h, 0);

    int top = option_find_int(options, "top", 1);

    char *label_list = option_find_str(options, "labels", 0);
    char *name_list = option_find_str(options, "names", label_list);
    char **names = get_labels(name_list);

    int *indexes = calloc(top, sizeof(int));

    if(!cap) error("Couldn't connect to webcam.\n");
    float fps = 0;
    int i;

    while(1){
        struct timeval tval_before, tval_after, tval_result;
        gettimeofday(&tval_before, NULL);

        image in = get_image_from_stream(cap);
        //image in_s = resize_image(in, net->w, net->h);
        image in_s = letterbox_image(in, net->w, net->h);

        float *predictions = network_predict(net, in_s.data);
        if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
        top_predictions(net, top, indexes);

        printf("\033[2J");
        printf("\033[1;1H");
        printf("\nFPS:%.0f\n",fps);

        int lh = in.h*.03;
        int toph = 3*lh;

        float rgb[3] = {1,1,1};
        for(i = 0; i < top; ++i){
            printf("%d\n", toph);
            int index = indexes[i];
            printf("%.1f%%: %s\n", predictions[index]*100, names[index]);

            char buff[1024];
            sprintf(buff, "%3.1f%%: %s\n", predictions[index]*100, names[index]);
            image label = get_label(alphabet, buff, lh);
            draw_label(in, toph, lh, label, rgb);
            toph += 2*lh;
            free_image(label);
        }

        show_image(in, base, 10);
        free_image(in_s);
        free_image(in);

        gettimeofday(&tval_after, NULL);
        timersub(&tval_after, &tval_before, &tval_result);
        float curr = 1000000.f/((long int)tval_result.tv_usec);
        fps = .9*fps + .1*curr;
    }
#endif
}

void run_classifier(int argc, char **argv)
{
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int ngpus;
    int *gpus = read_intlist(gpu_list, &ngpus, gpu_index);


    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int top = find_int_arg(argc, argv, "-t", 0);
    int clear = find_arg(argc, argv, "-clear");
    char *data = argv[3];
    char *cfg = argv[4];
    /*
     * weights  : .weights
     * filename : .qweights
     * layer_s  : .qparams
     * csv      : save outputs to csv format
     * */
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6] : 0;
    char *layer_s = (argc > 7) ? argv[7] : 0;
    char *csv = (argc > 8) ? argv[8] : 0;
    int layer = layer_s ? atoi(layer_s) : -1;
    if(0==strcmp(argv[2], "predict")) predict_classifier(data, cfg, weights, filename, top);
    else if(0==strcmp(argv[2], "train")) train_classifier(data, cfg, weights, gpus, 1, clear);
    else if(0==strcmp(argv[2], "fout")) file_output_classifier(data, cfg, weights, filename);
    else if(0==strcmp(argv[2], "try")) try_classifier(data, cfg, weights, filename, atoi(layer_s));
    else if(0==strcmp(argv[2], "demo")) demo_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "gun")) gun_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "threat")) threat_classifier(data, cfg, weights, cam_index, filename);
    else if(0==strcmp(argv[2], "test")) test_classifier(data, cfg, weights, layer);
    else if(0==strcmp(argv[2], "csv")) csv_classifier(data, cfg, weights);
    else if(0==strcmp(argv[2], "label")) label_classifier(data, cfg, weights);
    else if(0==strcmp(argv[2], "validmulti")) validate_classifier_multi(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid10")) validate_classifier_10(data, cfg, weights);
    else if(0==strcmp(argv[2], "validcrop")) validate_classifier_crop(data, cfg, weights);
    else if(0==strcmp(argv[2], "validfull")) validate_classifier_full(data, cfg, weights);

    else if(0==strcmp(argv[2], "prune")) prune_weights(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "tune")) tune_weights(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "tuned_qparams")) tuned_qparams(data, cfg, weights, filename, gpus, 1, clear, argv[2]);

    else if(0==strcmp(argv[2], "txt_to_bin")) txt_to_bin(data, cfg, weights, gpus, 1);
    else if(0==strcmp(argv[2], "train_transfer")) train_classifier_by_transfer_learning(data, cfg, weights, gpus, 1, clear);
    else if(0==strcmp(argv[2], "train_tiny")) train_tiny_imagenet_without_few_layers(data, cfg, weights, gpus, 1, clear);
    else if(0==strcmp(argv[2], "valid_without_xpose")) validate_classifier_without_transpose_info(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid_imgnet_without_xpose")) validate_imagenet_pretrained_classifier(data, cfg, weights);

    else if(0==strcmp(argv[2], "valid_pruned_int8")) validate_pruned_classifier_int8(data, cfg, weights, filename, layer_s, argv[2]);
    else if(0==strcmp(argv[2], "valid_both")) validate_classifier_single_quant(data, cfg, weights, filename, layer_s, argv[2]);
    else if(0==strcmp(argv[2], "valid_cmp")) compare_fp_and_int(data, cfg, weights, filename, layer_s, argv[2]);
    else if(0==strcmp(argv[2], "valid_csv")) save_into_csv(data, cfg, weights, filename, layer_s, csv, argv[2]);
    else if(0==strcmp(argv[2], "valid_selection")) validate_via_qsz_selection(data, cfg, weights, filename, layer_s, argv[2]);

    else if(0==strcmp(argv[2], "mmqp")) minmax_qparams_through_one_epoch(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "mmqp_without_xpose")) minmax_qparams_without_transpose_info(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "train_cluster")) train_qparams_per_cluster(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "batch_qparams")) batch_qparams(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "input_qparams")) input_qparams(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "train_staged")) train_qparams_staged(data, cfg, weights, gpus, 1, clear, argv[2]);

    else if(0==strcmp(argv[2], "train_seq")) train_classifier_seq(data, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "train_randseq")) train_classifier_with_random_and_sequential_dataset(data, cfg, weights, gpus, 1, clear);

    else if(0==strcmp(argv[2], "qat")) quantization_aware_training(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "qat_int4")) quantization_aware_training_int4(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "qat_int8")) quantization_aware_training_int8(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "qat_bmc4")) quantization_aware_training_with_batch_of_mixed_clusters_int4(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "qat_ctr_s4")) quantization_aware_training_with_single_cluster_info_int4(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "qat_ctr_s8")) quantization_aware_training_with_single_cluster_info_int8(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "qat_ctr_b4")) quantization_aware_training_with_batch_cluster_info_int4(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "qat_ctr_b8")) quantization_aware_training_with_batch_cluster_info_int8(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "qat_tiny_int4")) quantization_aware_training_of_tiny_imagenet_int4(data, cfg, weights, gpus, 1, clear, argv[2]);
    else if(0==strcmp(argv[2], "qat_tiny_int8")) quantization_aware_training_of_tiny_imagenet_int8(data, cfg, weights, gpus, 1, clear, argv[2]);

    else if(0==strcmp(argv[2], "valid")) validate_classifier_single(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid_imgnet")) validate_imagenet_classifier(data, cfg, weights);
    else if(0==strcmp(argv[2], "valid_int4")) validate_quantized_network_int4(data, cfg, weights, filename, argv[2]);
    else if(0==strcmp(argv[2], "valid_int8")) validate_quantized_network_int8(data, cfg, weights, filename, argv[2]);
    else if(0==strcmp(argv[2], "valid_ctr_s4")) validate_quantized_network_with_single_cluster_info_int4(data, cfg, argv[2]);
    else if(0==strcmp(argv[2], "valid_ctr_s8")) validate_quantized_network_with_single_cluster_info_int8(data, cfg, argv[2]);
}
