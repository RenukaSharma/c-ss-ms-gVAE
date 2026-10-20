

res_dir="/home1/renuka/AD/results_bayesian_nanoFibre/BS-128_baseline_BinClassification"
mkdir ${res_dir}

for i in {0..4}
do      
    for ratio_l in 0.2
    do 
        
        mkdir ${res_dir}/ratio_l_${ratio_l}
        mkdir ${res_dir}/ratio_l_${ratio_l}/run_${i}

        python baseline_binary_classifier.py nanofibre nanofibre_vae ${res_dir}/ratio_l_${ratio_l}/run_${i} /home1/renuka/nanofibre --ratio_known_normal 0.80 --ratio_known_outlier ${ratio_l} --lr 1e-04 --n_epochs 150 --batch_size 128 --pretrain False --seed ${i} --ratio_pollution 0 --lr_milestone 100;   
        
    done    
done

# res_dir="/home1/renuka/AD/results_bayesian_Malaria-patchwise/BS-128/with-SPnoise-no-pollution-inv-loss/baseline_BinCls"
res_dir="/home1/renuka/AD/results_bayesian_nanoFibre/BS-128_baseline_BinClassification"
mkdir ${res_dir}

for i in {0..4}
do  
    for ratio_l in 0.1
    do 
        
        mkdir ${res_dir}/ratio_l_${ratio_l}
        mkdir ${res_dir}/ratio_l_${ratio_l}/run_${i}

        python baseline_binary_classifier.py nanofibre nanofibre_vae ${res_dir}/ratio_l_${ratio_l}/run_${i} /home1/renuka/nanofibre --ratio_known_normal 0.90 --ratio_known_outlier ${ratio_l} --lr 1e-04 --n_epochs 150 --batch_size 128 --pretrain False --seed ${i} --ratio_pollution 0 --lr_milestone 100;   
        
    done 

    for ratio_l in 0.05
    do 
        
        mkdir ${res_dir}/ratio_l_${ratio_l}
        mkdir ${res_dir}/ratio_l_${ratio_l}/run_${i}

        python baseline_binary_classifier.py nanofibre nanofibre_vae ${res_dir}/ratio_l_${ratio_l}/run_${i} /home1/renuka/nanofibre --ratio_known_normal 0.1 --ratio_known_outlier ${ratio_l} --lr 1e-04 --n_epochs 150 --batch_size 128 --pretrain False --seed ${i} --ratio_pollution 0 --lr_milestone 100;   
        
    done 

    for ratio_l in 0.01
    do 
        
        mkdir ${res_dir}/ratio_l_${ratio_l}
        mkdir ${res_dir}/ratio_l_${ratio_l}/run_${i}

        python baseline_binary_classifier.py nanofibre nanofibre_vae ${res_dir}/ratio_l_${ratio_l}/run_${i} /home1/renuka/nanofibre --ratio_known_normal 0.1 --ratio_known_outlier ${ratio_l} --lr 1e-03 --n_epochs 150 --batch_size 128 --pretrain False --seed ${i} --ratio_pollution 0 --lr_milestone 100;   
        
    done 
    
done    
