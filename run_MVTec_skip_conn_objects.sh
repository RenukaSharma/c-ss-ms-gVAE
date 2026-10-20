for category in capsule hazelnut metal_nut pill screw toothbrush transistor zipper # bottle cable
do
    mkdir /home1/renuka/AD/results_UNet_bayesian_MVTec_alphaNbeta/MVTec-${category}_w_classifier
    res_dir="/home1/renuka/AD/results_UNet_bayesian_MVTec_alphaNbeta/MVTec-${category}_w_classifier/BS-32"
    mkdir ${res_dir}
    for n_known_outlier_classes in 1 
    do
        for i in {0..4} #6_with-recon-param # {0..4} done
        do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
            for ratio_l in 0.2 0 0.05 0.01 0.1 0.49
            do
                for val in 1e-1 1e-3 1e-5 # 1e-3 1e-2 1e-4 1e-1 # 1e-1 1e-5 1e-2 1e-3 1e-4  
                do
                    for recon_param in 1 0 0.5 2
                    do
                        for gamma in 1 0.5 0 2 # 1 2 4 # 6 8 10 1
                        do  
                            for eta in 1 # 4 8 10
                            do                             
                                for baseline in A # B C D E # A
                                do                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma} 
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                    python main.py mvtec mvtec_net ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/objects-with-novel-ano-in-test/${category} --ratio_known_outlier ${ratio_l} --lr 1e-05 --n_epochs 75 --batch_size 32 --weight_decay 0.5e-6 --pretrain False --ae_n_epochs 50 --ae_lr 1e-5 --ae_batch_size 128 --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --latent_param ${gamma} --recon_param ${recon_param} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 1 --ablation_type ${baseline} --category ${category} --lr_milestone 10 --lr_milestone 20; #  --lr_milestone 15
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

for category in bottle cable
do
    mkdir /home1/renuka/AD/results_UNet_bayesian_MVTec_alphaNbeta/MVTec-${category}_w_classifier
    res_dir="/home1/renuka/AD/results_UNet_bayesian_MVTec_alphaNbeta/MVTec-${category}_w_classifier/BS-128"
    mkdir ${res_dir}
    for n_known_outlier_classes in 1 
    do
        for i in {0..4} #6_with-recon-param # {0..4} done
        do
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}  
            for ratio_l in 0.2 0 0.05 0.01 0.1 0.49
            do
                for val in 1e-1 1e-3 1e-5 # 1e-3 1e-2 1e-4 1e-1 # 1e-1 1e-5 1e-2 1e-3 1e-4  
                do
                    for recon_param in 0 0.5 1 2
                    do
                        for gamma in 1 0.5 0 2 # 1 2 4 # 6 8 10 1
                        do  
                            for eta in 1 # 4 8 10
                            do                             
                                for baseline in A # B C D E # A
                                do                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma} 
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}                       
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}
                                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i}

                                    python main.py mvtec mvtec_net ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/recon_param_${recon_param}/gamma_${gamma}/eta_${eta}/val_${val}/baseline_${baseline}/run_${i} /home1/renuka/AD/mvtec_AD_dataset/objects-with-novel-ano-in-test/${category} --ratio_known_outlier ${ratio_l} --lr 1e-05 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_n_epochs 50 --ae_lr 1e-5 --ae_batch_size 128 --n_known_outlier_classes ${n_known_outlier_classes} --seed ${i} --latent_param ${gamma} --recon_param ${recon_param} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 1 --ablation_type ${baseline} --category ${category} --lr_milestone 10 --lr_milestone 20;   # --lr_milestone 10 --lr_milestone 20  --lr_milestone 15
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
