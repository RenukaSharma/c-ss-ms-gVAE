
res_dir="/home/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/no-noise_with0.1pollution_inv-loss-noveltest_May22/baseline-ssad-hybrid"
mkdir ${res_dir}
for i in 5 # {0..4}
do
    for n_known_outlier_classes in 1
    do
        for ratio_l in 0.2 # 0.1 0.05 0.01 0
        do
            mkdir ${res_dir}/ratio_l_${ratio_l}
            for kappa in 1
            do
                mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}
                mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i}

                python baseline_ssad.py malaria_dataset ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i} /home1/renuka/malaria_dataset/ORIGINAL_dataset/malaria/curated_malaria_dataset_manual_annotations_train_test --ratio_known_outlier ${ratio_l} --kernel rbf --kappa ${kappa} --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --ratio_pollution 0.1 --hybrid True --load_ae /home1/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/with-SPnoise-no-pollution-inv-loss/n_known_outlier_classes_1/ratio_l_${ratio_l}/recon_param_1/latent_param_0.5/eta_1/val_1e-1/baseline_A/run_1/model.tar;
            done
        done
    done      
done

res_dir="/home/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/no-noise_with0.1pollution_inv-loss-noveltest_May22/baseline-ssad-hybrid"
mkdir ${res_dir}
for i in {0..4}
do
    for n_known_outlier_classes in 1
    do
        for ratio_l in 0.1 0.05 0.01 0 # 0.2
        do
            mkdir ${res_dir}/ratio_l_${ratio_l}
            for kappa in 1
            do
                mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}
                                                    
                mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i}

                python baseline_ssad.py malaria_dataset ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i} /home1/renuka/malaria_dataset/ORIGINAL_dataset/malaria/curated_malaria_dataset_manual_annotations_train_test --ratio_known_outlier ${ratio_l} --kernel rbf --kappa ${kappa} --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --ratio_pollution 0.1 --hybrid True --load_ae /home1/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/with-SPnoise-no-pollution-inv-loss/n_known_outlier_classes_1/ratio_l_${ratio_l}/recon_param_1/latent_param_0.5/eta_1/val_1e-1/baseline_A/run_1/model.tar;
            done
        done
    done      
done

# ------ with load_ae=False ------

res_dir="/home/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/no-noise_with0.1pollution_inv-loss-noveltest_May22/baseline-ssad"
mkdir ${res_dir}
for i in {0..4}
do
    for n_known_outlier_classes in 1
    do
        for ratio_l in 0.2 # 0.1 0.05 0.01 0
        do
            mkdir ${res_dir}/ratio_l_${ratio_l}
            for kappa in 1
            do
                mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}
                                                    
                mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i}

                python baseline_ssad.py malaria_dataset ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i} /home1/renuka/malaria_dataset/ORIGINAL_dataset/malaria/curated_malaria_dataset_manual_annotations_train_test --ratio_known_outlier ${ratio_l} --kernel rbf --kappa ${kappa} --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --ratio_pollution 0.1 ;
            done
        done
    done      
done

res_dir="/home/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/no-noise_with0.1pollution_inv-loss-noveltest_May22/baseline-ssad"
mkdir ${res_dir}
for i in {0..4}
do
    for n_known_outlier_classes in 1
    do
        for ratio_l in 0.1 0.05 0.01 0 # 0.2
        do
            mkdir ${res_dir}/ratio_l_${ratio_l}
            for kappa in 1
            do
                mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}
                                                    
                mkdir ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i}

                python baseline_ssad.py malaria_dataset ${res_dir}/ratio_l_${ratio_l}/kappa_${kappa}/run_${i} /home1/renuka/malaria_dataset/ORIGINAL_dataset/malaria/curated_malaria_dataset_manual_annotations_train_test --ratio_known_outlier ${ratio_l} --kernel rbf --kappa ${kappa} --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --ratio_pollution 0.1 ;
            done
        done
    done      
done
