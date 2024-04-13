1. prepare test data.
  1.1 download data from url: ...
  1.2 unzip the test data.

2. calculate the de-biased mulit-model forecasts
  2.1 set path variables (in function "main_proc") in Station_FCST_BC.py
  2.2 run Station_FCST_BC.py to de-bias the raw forecast.

3. do the traditional weighted multi-model blending and improved weighted multi-model blending
  3.1 set path variables (in function "prepare_corrcoef" and function "main_proc") in Station_FCST_MMWB.py
  3.2 run Station_FCST_MMWB.py to blend multi-model de-biased forecasts

4. test the blended forecast's expection and STD estimated equation (Fig. 1)
  run N01.py

5. test the MAE estimate equation (Fig. 2)
  run N02.py

6. show 72H blended forecast's MAE variation curve (Fig. 3)
  run N03.py 
