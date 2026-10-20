for category in "carpet" # 0
    do
    res_dir="/home1/renuka/AD/results_bayesian_MVTec/MVTec-${category}-Jan8-final-res/BS-128/baseline-ssad-hybrid-withVarLearning"
    mkdir ${res_dir}
    for i in 6 # {0..4}
        do
        for n_known_outlier_classes in 1
            do
            for ratio_l in 0.2 0.1 0.05 0.01 0
                do
                mkdir ${res_dir}/ratio_l_${ratio_l}
                for kappa in 1
                    do
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}
                                                        
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i}

                    python baseline_ssad.py mvtec ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/carpet-with-novel-ano-in-test --ratio_known_outlier ${ratio_l} --kernel rbf --kappa ${kappa} --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --ratio_pollution 0.2 --hybrid True --load_ae /home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-carpet_inv-savingmodel/BS-1/n_known_outlier_classes_1/ratio_l_${ratio_l}/recon_param_1/gamma_1/eta_1/val_1e-1/baseline_A/run_0/model.tar;
                    done
                done
            done      
        done
    done

for i in 0
    do
    for category in "bottle" "cable" "capsule" "metal_nut" "pill" "screw" "toothbrush" "zipper" "hazelnut" "transistor"
        do
        mkdir /home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv
        mkdir /home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv/BS-1
        res_dir="/home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv/BS-1/ssad-hybrid-loadae"
        mkdir ${res_dir}
    
        for n_known_outlier_classes in 1
            do
            for ratio_l in 0.2 0
                do
                mkdir ${res_dir}/ratio_l_${ratio_l}
                for kappa in 1
                    do
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}
                                                        
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i}

                    python baseline_ssad.py mvtec ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/objects-with-novel-ano-in-test/${category} --ratio_known_outlier ${ratio_l} --kernel rbf --kappa ${kappa} --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --ratio_pollution 0.2 --hybrid True --load_ae /home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv-savingmodel/BS-1/n_known_outlier_classes_1/ratio_l_${ratio_l}/recon_param_1/gamma_1/eta_1/val_1e-1/baseline_A/run_0/model.tar;
                    done
                done
            done      
        done
    done


for i in 0
    do
    for category in "wood" #   "grid" "leather" "tile"
        do
        mkdir /home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/
        mkdir /home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/BS-1
        res_dir="/home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/BS-1/ssad-hybrid-loadae"
        mkdir ${res_dir}
    
        for n_known_outlier_classes in 1
            do
            for ratio_l in 0.2 0
                do
                mkdir ${res_dir}/ratio_l_${ratio_l}
                for kappa in 1
                    do
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}
                                                        
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i}

                    python baseline_ssad.py mvtec ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/${category}-with-novel-ano-in-test --ratio_known_outlier ${ratio_l} --kernel rbf --kappa ${kappa} --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --ratio_pollution 0.2 --hybrid True --load_ae /home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-carpet_inv-savingmodel/BS-1/n_known_outlier_classes_1/ratio_l_${ratio_l}/recon_param_1/gamma_1/eta_1/val_1e-1/baseline_A/run_0/model.tar;
                    done
                done
            done      
        done
    done

for i in {0..4}
    do
    for category in "bottle" "cable" "capsule" "metal_nut" "pill" "screw" "toothbrush" "zipper" "hazelnut" "transistor"
        do
        mkdir /home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv
        mkdir /home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv/BS-1
        res_dir="/home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv/BS-1/ssad"
        mkdir ${res_dir}
    
        for n_known_outlier_classes in 1
            do
            for ratio_l in 0.2 0
                do
                mkdir ${res_dir}/ratio_l_${ratio_l}
                for kappa in 1
                    do
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}
                                                        
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i}

                    python baseline_ssad.py mvtec ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/objects-with-novel-ano-in-test/${category} --ratio_known_outlier ${ratio_l} --kernel rbf --kappa ${kappa} --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --ratio_pollution 0.2;
                    done
                done
            done      
        done
    done


for i in {0..4}
    do
    for category in "grid" "leather" "tile" "wood"
        do
        mkdir /home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/
        mkdir /home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/BS-1
        res_dir="/home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/BS-1/ssad"
        mkdir ${res_dir}
    
        for n_known_outlier_classes in 1
            do
            for ratio_l in 0.2 0
                do
                mkdir ${res_dir}/ratio_l_${ratio_l}
                for kappa in 1
                    do
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}
                                                        
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i}

                    python baseline_ssad.py mvtec ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/${category}-with-novel-ano-in-test --ratio_known_outlier ${ratio_l} --kernel rbf --kappa ${kappa} --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --ratio_pollution 0.2;
                    done
                done
            done      
        done
    done

for i in {0..4}
    do
    for category in "carpet"
        do
        mkdir /home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/
        mkdir /home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/BS-1
        res_dir="/home1/renuka/AD/results_bayesian_MVTec/texture-categories/MVTec-${category}_inv/BS-1/ssad"
        mkdir ${res_dir}
    
        for n_known_outlier_classes in 1
            do
            for ratio_l in 0.2 0 0.01 0.05 0.1
                do
                mkdir ${res_dir}/ratio_l_${ratio_l}
                for kappa in 1
                    do
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}
                                                        
                    mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i}

                    python baseline_ssad.py mvtec ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/${category}-with-novel-ano-in-test --ratio_known_outlier ${ratio_l} --kernel rbf --kappa ${kappa} --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --ratio_pollution 0.2;
                    done
                done
            done      
        done
    done