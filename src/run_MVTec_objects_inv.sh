for i in 0 1 2 #6_with-recon-param # {0..4} done
do
    for category in "transistor" # "hazelnut" "bottle" "cable" "capsule" "metal_nut" "pill" "screw" "toothbrush" "zipper" "transistor"
    do
        mkdir /home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv-savingmodel
        res_dir="/home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv-savingmodel/BS-1"
        mkdir ${res_dir}
        for n_known_outlier_classes in 1 
        do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
            for ratio_l in 0
            do
                for val in 1e-1 # 1e-3 1e-2 1e-4 1e-1 # 1e-1 1e-5 1e-2 1e-3 1e-4  
                do
                    for recon_param in 1   
                    do
                        for gamma in 1  
                        do  
                            for eta in 1  
                            do                             
                                for baseline in D # later D at 0
                                do                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma} 
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                    python main.py mvtec cifar10_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/objects-with-novel-ano-in-test/${category} --ratio_known_outlier ${ratio_l} --lr 1e-05 --n_epochs 200 --batch_size 1 --weight_decay 0.5e-6 --pretrain False --ae_n_epochs 40 --ae_lr 1e-4 --ae_batch_size 128 --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --latent_param ${gamma} --recon_param ${recon_param} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0 --known_outlier_class 1 --ablation_type ${baseline} --category ${category};   # --lr_milestone 10 --lr_milestone 20  --lr_milestone 15
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

for i in 0 1 2 #6_with-recon-param # {0..4} done
do
    for category in "toothbrush" # "hazelnut" "bottle" "cable" "capsule" "metal_nut" "pill" "screw" "toothbrush" "zipper" "transistor"
    do
        mkdir /home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv-savingmodel
        res_dir="/home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv-savingmodel/BS-1"
        mkdir ${res_dir}
        for n_known_outlier_classes in 1 
        do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
            for ratio_l in 0.2
            do
                for val in 1e-1 # 1e-3 1e-2 1e-4 1e-1 # 1e-1 1e-5 1e-2 1e-3 1e-4  
                do
                    for recon_param in 1   
                    do
                        for gamma in 1  
                        do  
                            for eta in 1  
                            do                             
                                for baseline in D E A # latewr D at 0
                                do                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma} 
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                    python main.py mvtec cifar10_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/objects-with-novel-ano-in-test/${category} --ratio_known_outlier ${ratio_l} --lr 1e-05 --n_epochs 200 --batch_size 1 --weight_decay 0.5e-6 --pretrain False --ae_n_epochs 40 --ae_lr 1e-4 --ae_batch_size 128 --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --latent_param ${gamma} --recon_param ${recon_param} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0 --known_outlier_class 1 --ablation_type ${baseline} --category ${category};   # --lr_milestone 10 --lr_milestone 20  --lr_milestone 15
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

for i in 0 1 #6_with-recon-param # {0..4} done
do
    for category in "carpet" "grid" "leather" "wood" "tile"
    do
        mkdir /home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv-savingmodel
        res_dir="/home1/renuka/AD/results_bayesian_MVTec/object-categories/MVTec-${category}_inv-savingmodel/BS-1"
        mkdir ${res_dir}
        for n_known_outlier_classes in 1 
        do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
            for ratio_l in 0.2 0
            do
                for val in 1e-1 1e-5 # 1e-3 1e-2 1e-4 1e-1 # 1e-1 1e-5 1e-2 1e-3 1e-4  
                do
                    for recon_param in 1   
                    do
                        for gamma in 1  
                        do  
                            for eta in 1  
                            do                             
                                for baseline in A 
                                do                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma} 
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                    python main.py mvtec cifar10_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/${category}-with-novel-ano-in-test --ratio_known_outlier ${ratio_l} --lr 1e-05 --n_epochs 200 --batch_size 1 --weight_decay 0.5e-6 --pretrain False --ae_n_epochs 40 --ae_lr 1e-4 --ae_batch_size 128 --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --latent_param ${gamma} --recon_param ${recon_param} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0 --known_outlier_class 1 --ablation_type ${baseline} --category ${category};   # --lr_milestone 10 --lr_milestone 20  --lr_milestone 15
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done