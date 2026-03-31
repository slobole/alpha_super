##### Results:
* Add MAD (Mean Absoulute Deviation)
* Plots: Betta + Alpha
* Change name to vanilla backtest
* For debug only, for debug only, within a command: simulate a full trade, from signal to closing it -> it should plot OHLC bars + the indicator + signal + entry + exit. we should think what more feature we would like to add under this umbrella.
* Plots: Add Corona, and other crisis background colors.

##### Strats:
* For MR Strats, we can add also shorting.
* Entering Methods: IBS < 0.1, looking to CVAR
* Exiting methods: RSI < 2 or IBS > 0.9 or Stochastics(K=7, D=1, Smooth=3) > 0.8 ?
* Positioning: pos = round(150 / VIX.C)
* Implement https://zorro-project.com/manual/toc.php LPF.
* Use fisher transform on indicator ?