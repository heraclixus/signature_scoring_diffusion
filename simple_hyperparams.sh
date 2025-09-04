for batch_size in 16 32; do
    for num_samples in 8 16; do
        for dyadic in 1 2; do
            python simple_sigscore.py --batch_size $batch_size --num_samples $num_samples --dyadic_order $dyadic &> log_batch_${batch_size}_sample_${num_samples}_dy_${dyadic}.txt &  
        done
    done
done