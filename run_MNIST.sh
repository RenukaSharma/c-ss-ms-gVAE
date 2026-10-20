mkdir /home1/renuka/AD/results_bayesian_VAE_Sep8
res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep23/MNIST_rp-0.2"
# res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep23-ConvDeconv/MNIST_rp-0.2"
mkdir ${res_dir}
for n_known_outlier_classes in 1 # 5
do
    for i in {1..3}
    do 
        for gamma in 1 # 2 4 6 8 10
        do  
            for eta in 1 # 2 4 6 8 10
            do
                for ratio_l in 0.05 0.2 0.1 0.01 0
                do
                    for val in 1e-5 0.1 # 0.1 1e-3  # 0.001 0.01 
                    do
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val} 
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/run_${i} 

                        python main.py mnist mnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-4 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 1e-5 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 2 --ablation_type A;      # --lr_milestone 25 
                    done                   
                done
            done
        done
    done
done


python main_BinClassification.py mnist mnist_LeNet ../log/MNIST ../data --seed 0

res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep23/MNIST_rp-0.2/ablation-B"
mkdir ${res_dir}
for n_known_outlier_classes in 1 # 5
do
    for i in {1..3}
    do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}
        for gamma in 1 2 4 6 8 10
        do  
            for eta in 1 2 4 6 8 10
            do
                for ratio_l in 0.2 0.1 0.05 0.01 0
                do
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}        
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/run_${i}

                    python main.py mnist mnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-04 --n_epochs 75 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.001 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 2 --ablation_type B;  
                                                        
                done
            done
        done
    done
done

res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep23/MNIST_rp-0.2/ablation-C" ##sort it out
mkdir ${res_dir}
for n_known_outlier_classes in 1 # 5
do
    for i in {1..3}
    do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}
        for val in 0.1 1e-5 # 0.001 0.01 
        do  
            for gamma in 0 # 0.5 0.25 0
            do  
                for eta in 1
                do
                    for ratio_l in 0.2 0.1 0.05 0.01 0
                    do
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}
                        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/run_${i}

                        python main.py mnist mnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/val_${val}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-04 --n_epochs 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.001 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --eps ${val} --tau ${val} --delta ${val} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 2 --ablation_type C;   
                    done                          
                done
            done
        done
    done
done

### Following: to run

res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep20_GGnoise-new/MNIST_rp-0.2/ablation-D"
mkdir ${res_dir}
for n_known_outlier_classes in 1 # 5
do
    for i in {1..15}
    do 
        for gamma in 0 # 0.5 0.25 0
        do  
            for eta in 1
            do
                for ratio_l in 0.2 0.1 0.05 0.01 0
                do
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/run_${i}       
                                        
                    python main.py mnist mnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-02 --n_epochs 50 --batch_size 128  --lr_milestone 15 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.001 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 2 --ablation_type D;     
                                            
                done
            done
        done
    done
done

res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep23/MNIST_rp-0.2/ablation-E" ##sort it out
mkdir ${res_dir}
for n_known_outlier_classes in 1 # 5
do
    for i in {1..3}
    do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}
        for gamma in 100 1000 50 # 0.5 0.25 0
        do  
            for eta in 1 2 4 6 8 10
            do
                for ratio_l in 0.2 0.1 0.05 0.01 0
                do
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                    
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/run_${i}   
                    python main.py mnist mnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-03 --n_epochs 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.001 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --n_jobs_dataloader 0 --ratio_known_normal 0 --ratio_pollution 0.2 --known_outlier_class 2 --ablation_type E;  
                done     
            done
        done
    done
done

res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep8/MNIST_rp-0.2/ablation-VAE" 
mkdir ${res_dir}
for n_known_outlier_classes in 1
do
    for i in {1..15}
    do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}
        for gamma in 1 # 0.5 0.25 0
        do  
            for eta in 1
            do
                for ratio_l in 0.2 0.1 0.05 0.01 0
                do
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}            
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                    
                    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/run_${i}  

                    python main.py mnist mnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-02 --n_epochs 100 --lr_milestone 0 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.001 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --n_jobs_dataloader 0 --ratio_known_normal 0.1 --ratio_pollution 0.2 --known_outlier_class 2 --ablation_type VAE;                             
                done
            done
        done
    done
done

res_dir="/home1/renuka/AD/results_bayesian_VAE_Sep8/MNIST_rp-0.2/DeepSAD" 
mkdir ${res_dir}
for n_known_outlier_classes in 1
    do
    mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}
     
    for gamma in 1 # 0.5 0.25 0
    do  
    for eta in 1
    do
    for ratio_l in 0.05 0.2 0.1 0.01 0
        do
        mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}
        
            mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma} 
                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}
                for i in {1..15}
                do   
                mkdir ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/run_${i}  
                        python main.py mnist mnist_LeNet ${res_dir}/n_known_outlier_classes_${n_known_outlier_classes}/ratio_l_${ratio_l}/gamma_${gamma}/eta_${eta}/run_${i} ../data --ratio_known_outlier ${ratio_l} --lr 1e-02 --n_epochs 100 --lr_milestone 0 --batch_size 128 --weight_decay 0.5e-6 --pretrain False --ae_lr 0.001 --ae_n_epochs 20 --ae_lr_milestone 12 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 3 --n_known_outlier_classes ${n_known_outlier_classes} --seed 42 --recon_param ${gamma} --eta ${eta} --n_jobs_dataloader 0 --ratio_known_normal 0.1 --ratio_pollution 0.2 --known_outlier_class 2 --ablation_type DeepSAD;                             
                done
                done
            done
        done
    done