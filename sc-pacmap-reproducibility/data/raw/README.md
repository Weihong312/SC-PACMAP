# Raw data

Raw source files are not redistributed here. Download or obtain them from the cited providers and preserve the paths and filenames below. The preparation notebook checks every required file before processing.

## Critical Minerals in Ores

Source: https://ecat.ga.gov.au/geonetwork/srv/api/records/3c876198-5f78-4f6a-836e-da7f22c3333e

Place the source table at:

```text
cmio/CriticalMineralDepositsGeochemistry.csv
```

## North Batten

The preparation uses Northern Territory Geological Survey exploration-report files from the 2015, 2018, and 2019 campaigns. The supplied 2019 GR504 source is https://geoscience.nt.gov.au/gemis/ntgsjspui/handle/1/94531.

Place the files as follows:

```text
https://geoscience.nt.gov.au/gemis/ntgsjspui/handle/1/91634?utm_source=chatgpt.com
north_batten/2015/GR367_2015_GA_02_COLL2015A.txt
north_batten/2015/GR367_2015_GA_03_SURV2015A.txt
north_batten/2015/GR367_2015_GA_04_GEOL2015A.txt
north_batten/2015/GR367_2015_GA_06_ASS2015A.txt
north_batten/2015/GR367_2015_GA_10_LithCodes.csv

north_batten/2018/GR321_2018_GA_02_NBA_NTCL4_COLL2018A.txt
north_batten/2018/GR321_2018_GA_03_NBA_NTDS4_SURV2018A.txt
north_batten/2018/GR321_2018_GA_04_NBA_NTDL4_GEO2018A.txt
north_batten/2018/GR321_2018_GA_06_NBA_NTDG4_ASS2018A_1.txt
north_batten/2018/GR321_2018_GA_10_NBA_LithCodes.csv

north_batten/2019/GR504_2019_A_04_DrillCollars_revised.txt
north_batten/2019/GR504_2019_A_08_Survey_revised.txt
north_batten/2019/GR504_2019_A_07_Geol_revised.txt
north_batten/2019/GR504_2019_A_06_Assay_revised.txt
north_batten/2019/GR504_2019_A_12_LithCodes_revised.CSV
```

Run `data_preparation/prepare_datasets.ipynb` from the repository root. It writes:

```text
processed/cmio_sc_pacmap_ready.csv
processed/north_batten_sc_pacmap_ready.csv
```
