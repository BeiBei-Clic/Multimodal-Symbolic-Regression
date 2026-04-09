# PMLB LSO Batch Inference

以下命令默认在仓库根目录执行，且 Python 环境已经激活。

`python experiments/pmlb/pmlb_batch_inference.py --model_path ./weights/snip-e2e-sr.pth --device cuda:0 --dataset_limit 1 --max_rows 32 --max_input_points 32 --beam_size 1 --lso_pop_size 4 --lso_max_iteration 2 --noise_strength 0 --output_csv ./experiments/pmlb/results/pmlb_batch_inference_noise_0_smoke.csv`
轻量无噪声自检命令，只跑前 1 个数据集，并缩小数据量、采样点数、种群大小和迭代次数，用来验证 LSO 批量推理、续跑和 CSV 落盘能否跑通。

`python experiments/pmlb/pmlb_batch_inference.py --model_path ./weights/snip-e2e-sr.pth --device cuda:0 --beam_size 2 --lso_optimizer gwo --lso_pop_size 50 --lso_max_iteration 80 --lso_stop_r2 0.99`
完整无噪声批跑命令，处理全部符合条件的 PMLB 回归数据集，默认输出到 `experiments/pmlb/results/pmlb_batch_inference_noise_0.csv`。

`python experiments/pmlb/pmlb_batch_inference.py --model_path ./weights/snip-e2e-sr.pth --device cuda:0 --beam_size 2 --lso_optimizer gwo --lso_pop_size 50 --lso_max_iteration 30 --lso_stop_r2 0.99 --noise_strength 0.1`
带噪声实验示例命令，使用乘性高斯噪声并默认输出到 `experiments/pmlb/results/pmlb_batch_inference_noise_0.1.csv`。
