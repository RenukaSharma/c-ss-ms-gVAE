# full supervision
res_dir="/home/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/no-noise_0.1-pollution_inv-loss-noveltest_w_classifier_corrected_bothpositive"
mkdir ${res_dir}
for n_known_outlier_classes in 1 
do
    for i in {1..5} # 
    do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
        for val in 1e-3 1e-5 1e-1 # 1e-1 1e-5 1e-2 1e-3 1e-4  
        do
            for recon_param in 1 0.5 2 # 1 2 4 # 6 8 10 1
            do  
                for latent_param in 1 0.5 2 # 0.5 # 2 # 1 0.1  # 0.1 1
                do
                    for classifier_coeff in 5.0 10.0 1.0 0.5 2.0
                    do
                        for eta in 1 2 4 # 5 10 # 4 8 10
                        do
                            for ratio_l in 0.2 0.1 0.05 0.01 0 0.49 # 0.05 0.01 0 # 0.2 0.1 0.05 0.01 0
                            do 
                                for baseline in A # B C D E A # VAE
                                do                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param} 
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}/eta_${eta}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}/eta_${eta}/val_${val}                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}/eta_${eta}/val_${val}/baseline_${baseline}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                    python main.py malaria_dataset cifar10_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} /home1/renuka/malaria_dataset/ORIGINAL_dataset/malaria/curated_malaria_dataset_manual_annotations_train_test --ratio_known_outlier ${ratio_l} --lr 1e-05 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --recon_param ${recon_param} --latent_param ${latent_param} --classifier_coeff ${classifier_coeff} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 4 --ratio_known_normal 0.5 --ratio_pollution 0.1 --known_outlier_class 1 --ablation_type ${baseline} --lr_milestone 10;   # --lr_milestone 40 --lr_milestone 20  --lr_milestone 15
                                done
                            done
                        done
                    done
                done 
            done
        done
    done
done

res_dir="/home/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/no-noise_0.1-pollution_inv-loss-noveltest_w_classifier_corrected_posneg"
mkdir ${res_dir}
for n_known_outlier_classes in 1 
do
    for i in {6..7} # 
    do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
        for val in 1e-1 # 1e-1 1e-5 1e-2 1e-3 1e-4  
        do
            for recon_param in 1 0.5 # 1 2 4 # 6 8 10 1
            do  
                for latent_param in 1 0.5 # 0.5 # 2 # 1 0.1  # 0.1 1
                do
                    for classifier_coeff in 5.0 1.0 2.0
                    do
                        for eta in 1 # 5 10 # 4 8 10
                        do
                            for ratio_l in 0.2 0.1 0.05 0.01 0 0.49 # 0.05 0.01 0 # 0.2 0.1 0.05 0.01 0
                            do 
                                for baseline in A # B C D E A # VAE
                                do                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param} 
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}/eta_${eta}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}/eta_${eta}/val_${val}                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}/eta_${eta}/val_${val}/baseline_${baseline}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                    python main.py malaria_dataset cifar10_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/classifier_coeff_${classifier_coeff}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} /home1/renuka/malaria_dataset/ORIGINAL_dataset/malaria/curated_malaria_dataset_manual_annotations_train_test --ratio_known_outlier ${ratio_l} --lr 1e-05 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --recon_param ${recon_param} --latent_param ${latent_param} --classifier_coeff ${classifier_coeff} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 4 --ratio_known_normal 0.5 --ratio_pollution 0.1 --known_outlier_class 1 --ablation_type ${baseline} --lr_milestone 10;   # --lr_milestone 40 --lr_milestone 20  --lr_milestone 15
                                done
                            done
                        done
                    done
                done 
            done
        done
    done
done

res_dir="/home/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/no-noise_0.1-pollution_inv-loss-noveltest_May23"
mkdir ${res_dir}
for n_known_outlier_classes in 1 
do
    for i in {1..2}
    do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
        for val in 1e-1 1e-3 1e-5 # 1e-1 1e-5 1e-2 1e-3 1e-4  
        do
            for recon_param in 0.5 1 2 # 1 2 4 # 6 8 10 1
            do  
                for latent_param in 0.5 1 2 # 1 0.1  # 0.1 1
                do
                    for eta in 1 4 # 5 10 # 4 8 10
                    do
                        for ratio_l in 0.2 0.1 0.05 0.01 0 # 0.05 0.01 0 # 0.2 0.1 0.05 0.01 0
                        do 
                            for baseline in A # B C D E A # VAE
                            do                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param} 
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/eta_${eta}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/eta_${eta}/val_${val}                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/eta_${eta}/val_${val}/baseline_${baseline}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                python main.py malaria_dataset cifar10_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} /home1/renuka/malaria_dataset/ORIGINAL_dataset/malaria/curated_malaria_dataset_manual_annotations_train_test --ratio_known_outlier ${ratio_l} --lr 1e-05 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --recon_param ${recon_param} --latent_param ${latent_param} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 4 --ratio_known_normal 0 --ratio_pollution 0.1 --known_outlier_class 1 --ablation_type ${baseline} --lr_milestone 10;   # --lr_milestone 40 --lr_milestone 20  --lr_milestone 15
                            done
                        done
                    done
                done 
            done
        done
    done
done

res_dir="/home/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/no-noise_0.1-pollution_inv-loss-noveltest_May23"
mkdir ${res_dir}
for n_known_outlier_classes in 1 
do
    for i in {6..10}
    do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
        for val in 0.1 # 1e-1 1e-5 1e-2 1e-3 1e-4  
        do
            for recon_param in 1 # 1 2 4 # 6 8 10 1
            do  
                for latent_param in 1 # 1 0.1  # 0.1 1
                do
                    for eta in 1  # 5 10 # 4 8 10
                    do
                        for ratio_l in 0.2 0.1 0.05 0.01 0 # 0.05 0.01 0 # 0.2 0.1 0.05 0.01 0
                        do 
                            for baseline in D # A # VAE
                            do                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param} 
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/eta_${eta}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/eta_${eta}/val_${val}                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/eta_${eta}/val_${val}/baseline_${baseline}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                python main.py malaria_dataset cifar10_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/latent_param_${latent_param}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} /home1/renuka/malaria_dataset/ORIGINAL_dataset/malaria/curated_malaria_dataset_manual_annotations_train_test --ratio_known_outlier ${ratio_l} --lr 1e-05 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --recon_param ${recon_param} --latent_param ${latent_param} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 4 --ratio_known_normal 0 --ratio_pollution 0.1 --known_outlier_class 1 --ablation_type ${baseline} --lr_milestone 10;   # --lr_milestone 40 --lr_milestone 20  --lr_milestone 15
                            done
                        done
                    done
                done 
            done
        done
    done
done
