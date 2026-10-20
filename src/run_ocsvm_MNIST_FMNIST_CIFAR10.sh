res_dir="/home1/renuka/AD/results_baselines_ocsvm_ssad/ocsvm/mnist"
mkdir ${res_dir}
for n_known_outlier_classes in 5
    do
    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}    
    for ratio_l in 0.01 0.05 0.1 0.2
        do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
        for nu in 0.1 0.2 0.5 1
            do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}
            for i in {1..3}
                do
                                        
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i}
                    python baseline_ocsvm.py mnist ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i} ../data --ratio_known_outlier ${ratio_l} --kernel rbf --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 0;
                done
            done
        done      
    done

###############################################

res_dir="/home1/renuka/AD/results_baselines_ocsvm_ssad/ocsvm/fmnist"
mkdir ${res_dir}
for n_known_outlier_classes in 5
    do
    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}    
    for ratio_l in 0.01 0.05 0.1 0.2
        do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
        for nu in 0.1 0.2 0.5 1
            do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}
            for i in {1..3}
                do
                    
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i}
                    python baseline_ocsvm.py fmnist ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i} ../data --ratio_known_outlier ${ratio_l} --kernel rbf --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 0;
                done
            done
        done      
    done

###############################################

res_dir="/home1/renuka/AD/results_baselines_ocsvm_ssad/ocsvm/cifar10"
mkdir ${res_dir}
for n_known_outlier_classes in 5
    do
    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}    
    for ratio_l in 0.01 0.05 0.1 0.2
        do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
        for nu in 0.1 0.2 0.5 1
            do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}
            for i in {1..3}
                do
                    
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i}
                    python baseline_ocsvm.py cifar10 ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i} ../data --ratio_known_outlier ${ratio_l} --kernel rbf --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 0;
                done
            done
        done      
    done

################# for ratio 0 ###################

res_dir="/home1/renuka/AD/results_baselines_ocsvm_ssad/ocsvm/mnist"
mkdir ${res_dir}
for n_known_outlier_classes in 5
    do
    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}    
    for ratio_l in 0
        do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
        for nu in 0.1 0.2 0.5 1
            do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}
            for i in {1..3}
                do
                                        
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i}
                    python baseline_ocsvm.py mnist ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i} ../data --ratio_known_outlier ${ratio_l} --kernel rbf --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 0;
                done
            done
        done      
    done

###############################################

res_dir="/home1/renuka/AD/results_baselines_ocsvm_ssad/ocsvm/fmnist_pollution_0-2"
mkdir ${res_dir}
for n_known_outlier_classes in 1 5
    do
    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}    
    for ratio_l in 0
        do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
        for nu in 0.1 0.2 0.5 1
            do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}
            for i in 1
                do
                    
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i}
                    python baseline_ocsvm.py fmnist ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i} ../data --ratio_known_outlier ${ratio_l} --kernel rbf --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 0 --ratio_pollution 0.2;
                done
            done
        done      
    done

###############################################

res_dir="/home1/renuka/AD/results_baselines_ocsvm_ssad/ocsvm/cifar10_pollution_0-1"
mkdir ${res_dir}
for n_known_outlier_classes in 5
    do
    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}    
    for ratio_l in 0
        do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
        for nu in 0.1 0.2 0.5 1
            do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}
            for i in {1..3}
                do                    
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i}
                    python baseline_ocsvm.py cifar10 ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i} ../data --ratio_known_outlier ${ratio_l} --kernel rbf --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 0 --ratio_pollution 0.1;
                done
            done
        done      
    done

########################### added Jul 2################
res_dir="/home1/renuka/AD/results_baselines_ocsvm_ssad/ocsvm/mnist_pollution_0-2"
mkdir ${res_dir}
for n_known_outlier_classes in 1 5
    do
    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}    
    for ratio_l in 0
        do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
        for nu in 0.1 0.2 0.5 1
            do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}
            for i in {1..3}
                do
                                        
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i}
                    python baseline_ocsvm.py mnist ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/nu_${nu}/run_${i} ../data --ratio_known_outlier ${ratio_l} --kernel rbf --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 0 --ratio_pollution 0.2;
                done
            done
        done      
    done
