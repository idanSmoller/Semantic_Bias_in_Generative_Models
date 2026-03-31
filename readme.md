to reproduce the multimodel experiments, run the following command:

```bash
python multi_model_runner.py --models sdxl sd15 sd21
```
(you can remove certain models if you don't want to run them)
(also feel free to change the professions in the script to your liking)

If you want to run a model comparison, you can run the following command:

```bash
python model_comparison.py --models sdxl sd21
```
 will compare the two models. note that you need to have the results of the multimodel experiments to run this script, so make sure to run the first command before running this one.



 to reproduce the text projection results, please run the following command:
```bash
python projection_text.py
```
(once again, feel free to change the professions in the script to your liking)