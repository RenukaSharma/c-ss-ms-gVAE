
python baseline_binary_classifier.py mvtec cifar10_LeNet /home1/renuka/AD/results_bayesian_MVTec/MVTec-carpet-Jan8-final-res/BS-128/baseline-supervised/run_1 /home1/renuka/AD/mvtec_AD_dataset/carpet-with-novel-ano-in-test --ratio_known_normal 0.8 --ratio_known_outlier 0.2 --ratio_pollution 0.2 --seed 128 --lr 1e-4 --n_epochs 150 --batch_size 128 --pretrain False --normal_class 0 --lr_milestone 100;

for category in "carpet" # 0
do
    res_dir="/home1/renuka/AD/results_bayesian_MVTec/MVTec-${category}-Jan8-final-res/BS-128/baseline-supervised-ratio-known-normal-varying"
    mkdir ${res_dir}
    
    for i in {0..4}
    do      
        for ratio_l in 0.01
        do 
            
            mkdir ${res_dir}/ratio_l_${ratio_l}
            mkdir ${res_dir}/ratio_l_${ratio_l}/run_${i}

            python baseline_binary_classifier.py mvtec cifar10_LeNet ${res_dir}/ratio_l_${ratio_l}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/carpet-with-novel-ano-in-test --ratio_known_normal 0.1 --ratio_known_outlier ${ratio_l} --lr 1e-03 --n_epochs 150 --batch_size 128 --pretrain False --seed ${i} --ratio_pollution 0.2 --lr_milestone 100;   
            
        done 

        for ratio_l in 0.2
        do 
            
            mkdir ${res_dir}/ratio_l_${ratio_l}
            mkdir ${res_dir}/ratio_l_${ratio_l}/run_${i}

            python baseline_binary_classifier.py mvtec cifar10_LeNet ${res_dir}/ratio_l_${ratio_l}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/carpet-with-novel-ano-in-test --ratio_known_normal 0.80 --ratio_known_outlier ${ratio_l} --lr 1e-04 --n_epochs 150 --batch_size 128 --pretrain False --seed ${i} --ratio_pollution 0.2 --lr_milestone 100;   
            
        done    

        for ratio_l in 0.1
        do 
            
            mkdir ${res_dir}/ratio_l_${ratio_l}
            mkdir ${res_dir}/ratio_l_${ratio_l}/run_${i}

            python baseline_binary_classifier.py mvtec cifar10_LeNet ${res_dir}/ratio_l_${ratio_l}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/carpet-with-novel-ano-in-test --ratio_known_normal 0.90 --ratio_known_outlier ${ratio_l} --lr 1e-04 --n_epochs 150 --batch_size 128 --pretrain False --seed ${i} --ratio_pollution 0.2 --lr_milestone 100;   
            
        done 

        for ratio_l in 0.05
        do 
            
            mkdir ${res_dir}/ratio_l_${ratio_l}
            mkdir ${res_dir}/ratio_l_${ratio_l}/run_${i}

            python baseline_binary_classifier.py mvtec cifar10_LeNet ${res_dir}/ratio_l_${ratio_l}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/carpet-with-novel-ano-in-test --ratio_known_normal 0.1 --ratio_known_outlier ${ratio_l} --lr 1e-04 --n_epochs 150 --batch_size 128 --pretrain False --seed ${i} --ratio_pollution 0.2 --lr_milestone 100;   
            
        done 

        
    done    
done


for i in {0..4}
do
    for category in "bottle" "cable" "capsule" "metal_nut" "pill" "screw" "toothbrush" "zipper" "hazelnut" "transistor"
    do
        res_dir="/home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv/BS-1/baseline-supervised-ratio-known-normal-0"
        mkdir ${res_dir}
        
        for ratio_l in 0.2
        do 
            
            mkdir ${res_dir}/ratio_l_${ratio_l}
            mkdir ${res_dir}/ratio_l_${ratio_l}/run_${i}

            python baseline_binary_classifier.py mvtec cifar10_LeNet ${res_dir}/ratio_l_${ratio_l}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/objects-with-novel-ano-in-test/${category} --ratio_known_normal 0 --ratio_known_outlier ${ratio_l} --lr 1e-04 --n_epochs 150 --batch_size 1 --pretrain False --seed ${i} --ratio_pollution 0.2 --lr_milestone 100;   
            
        done          
    done    
done

for i in {0..4}
do
    for category in "grid" "leather" "tile" "wood"
    do
        mkdir /home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/
        mkdir /home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/BS-1
        res_dir="/home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/BS-1/baseline-supervised-ratio-known-normal-0"
        mkdir ${res_dir}
        
        for ratio_l in 0.2
        do 
            
            mkdir ${res_dir}/ratio_l_${ratio_l}
            mkdir ${res_dir}/ratio_l_${ratio_l}/run_${i}

            python baseline_binary_classifier.py mvtec cifar10_LeNet ${res_dir}/ratio_l_${ratio_l}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/${category}-with-novel-ano-in-test --ratio_known_normal 0 --ratio_known_outlier ${ratio_l} --lr 1e-04 --n_epochs 150 --batch_size 1 --pretrain False --seed ${i} --ratio_pollution 0.2 --lr_milestone 100;   
            
        done          
    done    
done
