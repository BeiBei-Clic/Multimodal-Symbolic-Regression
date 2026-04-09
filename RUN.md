# PMLB LSO Batch Inference

以下命令默认在仓库根目录执行，且 Python 环境已经激活。

`python experiments/pmlb/pmlb_batch_inference.py --model_path ./weights/snip-e2e-sr.pth --device cuda:0 --dataset_limit 2 --max_rows 64 --max_input_points 64 --beam_size 1 --lso_pop_size 8 --lso_max_iteration 5 --output_csv ./experiments/pmlb/results/pmlb_batch_inference_smoke.csv`
轻量自检命令，只跑前 2 个数据集，并缩小数据量、采样点数、种群大小和迭代次数，用来验证 LSO 批量推理与 CSV 落盘能否跑通。

`python experiments/pmlb/pmlb_batch_inference.py --model_path ./weights/snip-e2e-sr.pth --device cuda:0 --beam_size 2 --lso_optimizer gwo --lso_pop_size 50 --lso_max_iteration 80 --lso_stop_r2 0.99 --output_csv ./experiments/pmlb/results/pmlb_batch_inference.csv`
完整批跑命令，处理全部符合条件的 PMLB 回归数据集，并显式暴露 LSO 进化算法超参数；`max_rows=200` 和 `max_input_points=200` 继续使用脚本默认值。
