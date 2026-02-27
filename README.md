![PymooLab Header](app.png)

# PymooLab

PymooLab is an open-source visual analytics framework for multi-objective optimization, with integrated benchmark orchestration, local plugin discovery, and LLM-assisted artifact formulation.

This document is a persistent technical map intended to support the upcoming full PDFLaTeX manual, with emphasis on what the codebase currently contains and how new problems, metrics, and operators are created.

## Contact

- Principal contact: `santostf+metibr@ufop.edu.br`

## Environment Baseline

- Recommended Python: `3.11.x`
- Backend policy: CPU by default, JAX acceleration when requested and available

## Requirements (Curated)

### Base Runtime

```txt
numpy>=2.3,<3
pymoo>=0.6.1,<0.7
scipy>=1.16,<2
PySide6>=6.10,<7
qt-material>=2.17,<3
qt-material-icons>=0.4,<1
matplotlib>=3.10,<4
psutil>=7.2,<8
```

### Optional Profiles

- GPU (JAX acceleration):
  - `jax>=0.9,<1`
  - `jaxlib>=0.9,<1`
- LLM generation:
  - `anthropic>=0.76,<1`
- ML-dependent algorithms:
  - `scikit-learn>=1.8,<2`
- Development:
  - `pytest>=9,<10`

### Installation Examples

```bash
pip install numpy pymoo scipy PySide6 qt-material qt-material-icons matplotlib psutil
pip install jax jaxlib              # optional GPU/JAX profile
pip install anthropic               # optional LLM profile
pip install scikit-learn            # optional ML-dependent algorithms
pip install pytest                  # optional development profile
```

## Implemented Catalog (Names Only)

- Snapshot date: `2026-02-27`
- Algorithms (`285`): `ABSAEA`, `ACMMEA`, `AdaW`, `ADSAPSO`, `AENSGAII`, `AFSEA`, `AGEII`, `AGSEA`, `AMGPSL`, `ANSGAIII`, `APSEA`, `ARMOEA`, `AVGSAEA`, `BCEIBEA`, `BCEMOEAD`, `BiCo`, `BiGE`, `BLEAQII`, `BLSAEA`, `C3M`, `CAEAD`, `CAMOEA`, `CCGDE3`, `CCMO`, `cDPEA`, `CGLP`, `CLIA`, `CMaDPPs`, `CMDEIPCM`, `CMEGL`, `CMME`, `CMMO`, `CMOBR`, `CMOCSO`, `CMODEFTR`, `CMODRL`, `CMOEA_CD`, `CMOEAD`, `CMOEAMS`, `CMOEAMSG`, `CMOEBOD`, `CMOEMT`, `CMOES`, `CMOQLMT`, `CMOSMA`, `CNSDEDVC`, `CoMMEA`, `CPSMOEA`, `CSEA`, `CSEMT`, `CTSEA`, `DAEA`, `DBEMTO`, `DCNSGAIII`, `DEAGNG`, `DGEA`, `DirHVEI`, `DISK`, `DISK_2025`, `DISKplus`, `DKCA`, `DMMOEA`, `DMOEAeC`, `dMOPSO`, `DNNSGAII`, `DPCPRA`, `DPPPS`, `DPVAPS`, `DRLOSEMCMO`, `DRLSAEA`, `DSPCMDE`, `DSSEA`, `DVCEA`, `DWU`, `EAGMOEAD`, `EDNARMOEA`, `EFRRR`, `EIMEGO`, `EM_SAEA`, `EMCMMS`, `EMCMO`, `EMMOEA`, `eMOEA`, `EMOSKT`, `EMyOC`, `ENSMOEAD`, `ESBCEO`, `FDV`, `FLEA`, `FRCGM`, `GCNMOEA`, `GDE3`, `GFMMOEA`, `GLMO`, `gNSGAII`, `GPSOM`, `GrEA`, `GWASFGA`, `HEA`, `HeEMOEA`, `HHCMMEA`, `hpaEA`, `HREA`, `HypE`, `IBEA`, `ICMA`, `IDBEA`, `IMCMOEAD`, `IMMOEA`, `IMMOEAD`, `IMTCMO`, `IMTCMO_BS`, `ISIBEA`, `Izui`, `KLEA`, `KLNSGAII`, `KnEA`, `KRVEA`, `KTA2`, `KTS`, `LCMEA`, `LCSA`, `LDSAF`, `LERD`, `LMEA`, `LMOCSO`, `LMOEADS`, `LMPFE`, `LRMOEA`, `LSMOF`, `MaOEACSS`, `MaOEADDFC`, `MaOEAIGD`, `MaOEAIT`, `MaOEARD`, `MCCMO`, `MCEAD`, `MFFS`, `MFOSPEA2`, `MGCEA`, `MGSAEA`, `MMEAPSL`, `MMEAWI`, `MMOPSO`, `MO_Ring_PSO_SCD`, `MOBCA`, `MOCell`, `MOCGDE`, `MOCMA`, `MOEACKF`, `MOEAD2WA`, `MOEADAWA`, `MOEADCMA`, `MOEADCMT`, `MOEADD`, `MOEADDAE`, `MOEADDCWV`, `MOEADDE`, `MOEADDQN`, `MOEADDRA`, `MOEADDU`, `MOEADDYTS`, `MOEADEGO`, `MOEADFRRMAB`, `MOEADM2M`, `MOEADMRDL`, `MOEADPaS`, `MOEADPFE`, `MOEADSTM`, `MOEADUR`, `MOEADURAW`, `MOEADVA`, `MOEADVOV`, `MOEAIGDNS`, `MOEANZD`, `MOEAPC`, `MOEAPSL`, `MOEARE`, `MOEGS`, `MOL2SMEA`, `MOMBIII`, `MOMFEA`, `MOMFEAII`, `MOMFEASADE`, `MONAS`, `MOPSO`, `MOSD`, `MPAES`, `MPMMEA`, `MPSOD`, `MSCEA`, `MSCMO`, `MSEA`, `MSKEA`, `MSOPSII`, `MTCMO`, `MTDEMKTA`, `MTEADDN`, `MTS`, `MultiObjectiveEGO`, `MyODEMR`, `NBLEA`, `NMPSO`, `NNDREAMO`, `NNIA`, `NRVMOEA`, `NSBiDiCo`, `NSGAIIARSBX`, `NSGAIIconflict`, `NSGAIIDTI`, `NSGAIIIEHVI`, `NSGAIISDR`, `NSLS`, `NUCEA`, `onebyoneEA`, `OSPNSDE`, `ParEGO`, `PBNSGAIII`, `PBRVEA`, `PCSAEA`, `PEA`, `PEAplus`, `PeEA`, `PESAII`, `PICEAg`, `PIEA`, `PIMD`, `PMMOEA`, `POCEA`, `PPS`, `PRDH`, `PREA`, `REMO`, `RGA_M1_2`, `RGA_M2_2`, `RMMEDA`, `RMOEADVA`, `RPDNSGAII`, `RPEA`, `RSEA`, `RVEAa`, `RVEAiGNG`, `S3CMAES`, `SAMOEA_TL2M`, `SCDAS`, `SCEA`, `SECSO`, `SFADE`, `SGEA`, `SGECF`, `SIBEA`, `SIBEAkEMOSS`, `SLMEA`, `SMEA`, `SMOA`, `SMPSO`, `SMSEGO`, `SNSGAII`, `SparseEA`, `SparseEA2`, `SPEA2SDE`, `SPEAR`, `SRA`, `SSCEA`, `SSDE`, `SSW`, `SSW_RDPA`, `SVRNSGAII`, `tDEA`, `tDEACPBI`, `TEA`, `TELSO`, `TiGE2`, `ToP`, `TPCMaO`, `TriMOEATAR`, `TSNSGAII`, `TSSparseEA`, `TSTI`, `Two_Arch2`, `URCMO`, `VaEA`, `WASFGA`, `WOF`, `WVMOEAP`
....................................................................................................
- Metrics (`25`): `CPF`, `DeltaP`, `DM`, `Feasible_rate`, `GD`, `HV`, `IGD`, `IGDp`, `IGDX`, `Lower_level_Min_value`, `Mean_HV`, `Mean_IGD`, `Min_value`, `PD`, `Spacing`, `Spread`, `Task1_HV`, `Task1_IGD`, `Task1_Min_value`, `Task2_HV`, `Task2_IGD`, `Task2_Min_value`, `Upper_level_Min_value`, `Worst_HV`, `Worst_IGD`
....................................................................................................
- Operators (`38`): `AgeBasedTournamentSelection`, `BFM`, `BinaryRandomSampling`, `BinomialCrossover`, `BitflipMutation`, `BX`, `ChoiceRandomMutation`, `DEX`, `EdgeRecombinationCrossover`, `ERX`, `ExponentialCrossover`, `FloatRandomSampling`, `GaussianMutation`, `GM`, `HalfUniformCrossover`, `HUX`, `IntegerRandomSampling`, `InversionMutation`, `LatinHypercubeSampling`, `LHS`, `NoCrossover`, `NoMutation`, `OrderCrossover`, `ParentCentricCrossover`, `PCX`, `PermutationRandomSampling`, `PM`, `PointCrossover`, `PolynomialMutation`, `RandomSelection`, `SBX`, `SimulatedBinaryCrossover`, `SinglePointCrossover`, `SPX`, `TournamentSelection`, `TwoPointCrossover`, `UniformCrossover`, `UX`
....................................................................................................
- Problem families (`41`): `BBOB`, `BT`, `CF`, `DASCMOP`, `DOC`, `DSMOP`, `DTLZ`, `FCP`, `FDA`, `GLSMOP`, `IMMOEA`, `IMOP`, `LIRCMOP`, `LRMOP`, `LSCM`, `LSMMOP`, `LSMOP`, `MAF`, `MAOPP`, `MMF`, `MMMOP`, `MOEADDE`, `MOEADM2M`, `MULTITASKING_MOPS`, `MULTITASKING_SOPS`, `MW`, `REALWORLD_MOPS`, `RMMEDA`, `RWMOP`, `SDC`, `SIMPLE_SOPS`, `SMD`, `SMMOP`, `SMOP`, `TP`, `UF`, `VNT`, `WFG`, `ZCAT`, `ZDT`, `ZXH_CF`

## Manual Continuity Note

This map is intentionally exhaustive and should be preserved as the baseline for the forthcoming PDFLaTeX manual focused on:
- Existing project capabilities and architecture.
- How to create new optimization problems.
- How to create and validate metrics.
- How to create and integrate operators.

If the codebase changes, regenerate this map to keep the manual source-of-truth synchronized.
