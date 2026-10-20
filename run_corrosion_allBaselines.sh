
for category in "carpet" # 0
do
    mkdir /home1/renuka/AD/results_bayesian_MVTec/MVTec-${category}-Jan16
    res_dir="/home1/renuka/AD/results_bayesian_corrosion_BS-128"
    mkdir ${res_dir}
    for n_known_outlier_classes in 1 
    do
        for i in {0..4} #6_with-recon-param # {0..4} done
        do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
            for ratio_l in 0 # 0.2 0.1 0.05 0.01 0 
            do
                for val in 1e-3 # 1e-3 1e-2 1e-4 1e-1 # 1e-1 1e-5 1e-2 1e-3 1e-4  
                do
                    for recon_param in 1 # 0 0.5 1 2
                    do
                        for gamma in 1 # 1 0.5 0 2 # 1 2 4 # 6 8 10 1
                        do  
                            for eta in 1 # 5 # 4 8 10
                            do                             
                                for baseline in A # B C D E # B C D E
                                do                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma} 
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                    python main.py mvtec cifar10_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} /home1/renuka/AD_CorrosionDataset/Corrosion_patches_dataset --ratio_known_outlier ${ratio_l} --lr 1e-05 --n_epochs 200 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_n_epochs 50 --ae_lr 1e-5 --ae_batch_size 128 --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --latent_param ${gamma} --recon_param ${recon_param} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 4 --ratio_known_normal 0 --ratio_pollution 0 --known_outlier_class 1 --ablation_type ${baseline} --category ${category} ;   # --lr_milestone 10 --lr_milestone 20  --lr_milestone 15
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

for normal_class in 3 # 0
do
    res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep26/fmnist_rp-0.2__normal_class-${normal_class}"
    mkdir ${res_dir}
    for n_known_outlier_classes in 1 
    do
        for i in 7
        do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
            for val in 0.01 # 1e-5 # 0.001 0.01 
            do
                for gamma in 2 # 2 4 # 6 8 10 1
                do  
                    for eta in 2 # 6 8 10
                    do
                        for ratio_l in 0 0.2 0.01 0.05 0.1 
                        do 
                            for baseline in A C B D E
                            do                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                python main.py fmnist fmnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-04 --lr_milestone 20 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.00001 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class ${normal_class} --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 1 --ablation_type ${baseline};   # --lr_milestone 20  --lr_milestone 10
                                
                            done
                        done                     
                    done
                done
            done
        done
    done
done

for normal_class in 3 # 0
do
    res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep26/fmnist_rp-0.2__normal_class-${normal_class}"
    mkdir ${res_dir}
    for n_known_outlier_classes in 1 
    do
        for i in 8
        do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
            for val in 0.1 # 1e-5 # 0.001 0.01 
            do
                for gamma in 2 # 2 4 # 6 8 10 1
                do  
                    for eta in 4 # 6 8 10
                    do
                        for ratio_l in 0 0.2 0.01 0.05 0.1 
                        do 
                            for baseline in A C B D E
                            do                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                python main.py fmnist fmnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-04 --lr_milestone 20 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.00001 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class ${normal_class} --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 1 --ablation_type ${baseline};   # --lr_milestone 20  --lr_milestone 10
                                
                            done
                        done                     
                    done
                done
            done
        done
    done
done

for normal_class in 3 # 0
do
    res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep26/fmnist_rp-0.2__normal_class-${normal_class}"
    mkdir ${res_dir}
    for n_known_outlier_classes in 1 
    do
        for i in 9
        do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
            for val in 0.01 # 1e-5 # 0.001 0.01 
            do
                for gamma in 2 # 2 4 # 6 8 10 1
                do  
                    for eta in 4 # 6 8 10
                    do
                        for ratio_l in 0 0.2 0.01 0.05 0.1 
                        do 
                            for baseline in A C B D E
                            do                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                python main.py fmnist fmnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-04 --lr_milestone 20 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.00001 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class ${normal_class} --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 1 --ablation_type ${baseline};   # --lr_milestone 20  --lr_milestone 10
                                
                            done
                        done                     
                    done
                done
            done
        done
    done
done

for normal_class in 3 # 0
do
    res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep26/fmnist_rp-0.2__normal_class-${normal_class}"
    mkdir ${res_dir}
    for n_known_outlier_classes in 1 
    do
        for i in 20
        do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
            for val in 1e-5 # 0.001 0.01 
            do
                for gamma in 1 # 2 4 # 6 8 10 1
                do  
                    for eta in 1 # 6 8 10
                    do
                        for ratio_l in 0 0.2 0.01 0.05 0.1 
                        do 
                            for baseline in E
                            do                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}                       
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}
                                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                python main.py fmnist fmnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-04 --lr_milestone 20 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.00001 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class ${normal_class} --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 1 --ablation_type ${baseline};   # --lr_milestone 20  --lr_milestone 10
                                
                            done
                        done                     
                    done
                done
            done
        done
    done
done
