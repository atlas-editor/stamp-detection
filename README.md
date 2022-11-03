

# Stamp Detection
Provides a way of analyzing official documents to detect the presence of stamps. The CLI component is used as follows.

```python
>>> python3 detect.py test.jpg
[[1751, 3021, 2105, 3375], [1538, 1740, 2245, 2068], [152, 85, 941, 369], [172, 2450, 1080, 2945]]
```

The stamp detection methods can also be used separately and fined-tuned if necessary. The analysis methods suite is in `stamp_feature_analysis.py`. Fine tuning can be done via tweaking the parameters in an instance of `AnalysisSettings` and/or `StampFeatureSettings` and passed to the `StampAnalysis` constructor which utilizes all analysis methods.

---
### Requirements
Python 3.8+ is required and also the following libraries:
- OpenCV

- NumPy

- PIL

See [`requirements.txt`](requirements.txt) for more info.

---
**Disclaimer**: the default constants are chosen empirically on a small set of documents to suit our purposes and may not work well for domains outside of official documents.

**References**:

​    [1] B. Micenkova and J. v. Beusekom. *Stamp Detection in Color Document Images*. 2011.

​    [2] P. Forczmanski and A. Markiewicz. *Stamps Detection and Classification Using Simple Features Ensemble*. 2015.
