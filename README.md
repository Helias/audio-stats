**Features of the dataset:**

spectrum,mean_frequency,peak_frequency,frequencies_std,amplitudes_cum_sum,  
mode_frequency,median_frequency,frequencies_q25,frequencies_q75,iqr,  
freqs_skewness,freqs_kurtosis,spectral_entropy,  
spectral_flatness,spectral_centroid,spectral_spread,spectral_rolloff,  
energy,rms,zcr,spectral_mean,spectral_rms,spectral_std,spectral_variance,  
meanfun,minfun,maxfun,meandom,mindom,maxdom,dfrange,modindex,bit_rate

**spectrum features**  
signal,mfcc,imfcc,bfcc,lfcc,lpc,lpcc,msrcc,ngcc,psrcc,plp,rplp,gfcc

**rows with the asterisk (\*) have all the features WITH the bit_rate, all rows without the asterisk (\*) have no bit_rate feature**

- 1: normal data without any normalization or resampling applied
- 2: data with resampled (lowered) bit rate
- 3: data with loud normalization applied

Here the synthetic technique where the bit_rate is not discriminating

| #         | #                                                               |
| --------- | --------------------------------------------------------------- |
| SYSTEM ID | A01-A06                                                         |
| Speakers  | LA_0069, LA_0070, LA_0071, LA_0072, LA_0073, LA_0074, LA_0075   |
| SYSTEM ID | A07, A08, A09, A10, A11, A12, A13, A14, A15, A16, A17, A18, A19 |
| Speakers  | LA_0012, LA_0013, LA_0047, LA_0023, LA_0038                     |

Accuracy of the related models with 10 iterations (the accuracy is the average of the 10 iterations).

| Model        | A01   | A02   | A03   | A04   | A05   | A06   | A07   | A08   | A09   | A10   | A11   | A12   | A13   | A14   | A15   | A16   | A17   | A18   | A19   |
| ------------ | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| **CART** 1   | 0.929 | 0.998 | 1.0   | 0.849 | 0.947 | 0.759 | 0.871 | 0.911 | 0.938 | 0.892 | 0.856 | 0.896 | 0.987 | 0.905 | 0.844 | 0.773 | 0.761 | 0.920 | 0.697 |
| **CART** 1\* | 0.951 | 0.998 | 1.0   | 0.890 | 0.937 | 0.790 | 0.957 | 0.951 | 0.972 | 0.954 | 0.938 | 0.908 | 0.996 | 0.954 | 0.896 | 0.795 | 0.776 | 0.911 | 0.737 |
| **CART** 2   | 0.930 | 1.0   | 1.0   | 0.855 | 0.954 | 0.790 | 0.847 | 0.905 | 0.957 | 0.874 | 0.874 | 0.914 | 0.984 | 0.905 | 0.831 | 0.801 | 0.822 | 0.917 | 0.694 |
| **CART** 2\* | 0.930 | 0.999 | 1.0   | 0.888 | 0.938 | 0.799 | 0.948 | 0.911 | 0.978 | 0.951 | 0.920 | 0.932 | 0.993 | 0.899 | 0.902 | 0.782 | 0.767 | 0.923 | 0.694 |
| **CART** 3   | 0.936 | 0.999 | 1.0   | 0.867 | 0.954 | 0.806 | 0.871 | 0.935 | 0.957 | 0.896 | 0.877 | 0.926 | 0.990 | 0.944 | 0.853 | 0.831 | 0.764 | 0.908 | 0.697 |
| **CART** 3\* | 0.972 | 1.0   | 1.0   | 0.892 | 0.959 | 0.799 | 0.975 | 0.932 | 0.954 | 0.966 | 0.957 | 0.899 | 0.990 | 0.981 | 0.935 | 0.798 | 0.795 | 0.908 | 0.730 |
| **SVM** 1    | 0.270 | 0.927 | 0.894 | 0.762 | 0.903 | 0.406 | 0.455 | 0.489 | 0.801 | 0.623 | 0.694 | 0.755 | 0.584 | 0.706 | 0.639 | 0.406 | 0.544 | 0.834 | 0.519 |
| **SVM** 1\*  | 0.856 | 0.970 | 0.807 | 0.803 | 0.908 | 0.711 | 0.932 | 0.868 | 0.865 | 0.954 | 0.932 | 0.810 | 0.957 | 0.914 | 0.874 | 0.813 | 0.596 | 0.828 | 0.498 |
| **SVM** 2    | 0.655 | 0.982 | 0.883 | 0.760 | 0.909 | 0.709 | 0.522 | 0.553 | 0.740 | 0.452 | 0.697 | 0.773 | 0.691 | 0.697 | 0.507 | 0.486 | 0.577 | 0.880 | 0.568 |
| **SVM** 2\*  | 0.843 | 0.831 | 0.823 | 0.823 | 0.884 | 0.681 | 0.908 | 0.810 | 0.825 | 0.914 | 0.923 | 0.776 | 0.914 | 0.840 | 0.840 | 0.611 | 0.498 | 0.880 | 0.461 |
| **SVM** 3    | 0.721 | 0.956 | 0.922 | 0.772 | 0.927 | 0.458 | 0.602 | 0.489 | 0.779 | 0.501 | 0.727 | 0.755 | 0.691 | 0.581 | 0.574 | 0.532 | 0.385 | 0.816 | 0.562 |
| **SVM** 3\*  | 0.874 | 0.973 | 0.922 | 0.816 | 0.920 | 0.741 | 0.932 | 0.785 | 0.825 | 0.948 | 0.920 | 0.810 | 0.975 | 0.938 | 0.892 | 0.819 | 0.611 | 0.856 | 0.590 |
| **LR** 1     | 0.737 | 0.992 | 0.886 | 0.759 | 0.873 | 0.712 | 0.724 | 0.798 | 0.752 | 0.651 | 0.672 | 0.785 | 0.914 | 0.862 | 0.593 | 0.651 | 0.636 | 0.874 | 0.581 |
| **LR** 1\*   | 0.933 | 0.919 | 0.877 | 0.877 | 0.860 | 0.704 | 0.960 | 0.944 | 0.923 | 0.969 | 0.975 | 0.905 | 0.975 | 0.932 | 0.883 | 0.813 | 0.596 | 0.847 | 0.617 |
| **LR** 2     | 0.737 | 0.993 | 0.875 | 0.761 | 0.868 | 0.729 | 0.547 | 0.721 | 0.712 | 0.629 | 0.752 | 0.767 | 0.917 | 0.807 | 0.611 | 0.614 | 0.654 | 0.886 | 0.541 |
| **LR** 2\*   | 0.866 | 0.850 | 0.863 | 0.859 | 0.915 | 0.703 | 0.917 | 0.856 | 0.785 | 0.908 | 0.935 | 0.865 | 0.960 | 0.865 | 0.828 | 0.785 | 0.596 | 0.892 | 0.645 |
| **LR** 3     | 0.795 | 0.997 | 0.919 | 0.783 | 0.927 | 0.720 | 0.639 | 0.795 | 0.782 | 0.636 | 0.721 | 0.743 | 0.914 | 0.840 | 0.629 | 0.691 | 0.602 | 0.813 | 0.535 |
| **LR** 3\*   | 0.952 | 0.926 | 0.931 | 0.886 | 0.923 | 0.736 | 0.981 | 0.862 | 0.788 | 0.990 | 0.966 | 0.859 | 0.978 | 0.966 | 0.926 | 0.847 | 0.629 | 0.859 | 0.645 |
| **KNN** 1    | 0.758 | 0.880 | 0.929 | 0.727 | 0.831 | 0.695 | 0.654 | 0.626 | 0.831 | 0.663 | 0.681 | 0.807 | 0.779 | 0.685 | 0.525 | 0.629 | 0.709 | 0.874 | 0.541 |
| **KNN** 1\*  | 0.924 | 0.778 | 0.912 | 0.863 | 0.753 | 0.707 | 0.941 | 0.892 | 0.905 | 0.954 | 0.938 | 0.850 | 0.941 | 0.914 | 0.865 | 0.779 | 0.587 | 0.813 | 0.614 |
| **KNN** 2    | 0.759 | 0.878 | 0.929 | 0.749 | 0.851 | 0.697 | 0.620 | 0.651 | 0.801 | 0.666 | 0.709 | 0.792 | 0.782 | 0.712 | 0.581 | 0.574 | 0.688 | 0.859 | 0.559 |
| **KNN** 2\*  | 0.864 | 0.781 | 0.913 | 0.837 | 0.725 | 0.692 | 0.923 | 0.819 | 0.896 | 0.917 | 0.948 | 0.877 | 0.941 | 0.837 | 0.816 | 0.730 | 0.599 | 0.868 | 0.605 |
| **KNN** 3    | 0.732 | 0.882 | 0.959 | 0.757 | 0.862 | 0.687 | 0.672 | 0.660 | 0.785 | 0.724 | 0.678 | 0.755 | 0.810 | 0.703 | 0.550 | 0.602 | 0.633 | 0.825 | 0.544 |
| **KNN** 3\*  | 0.934 | 0.768 | 0.946 | 0.858 | 0.751 | 0.707 | 0.944 | 0.804 | 0.792 | 0.966 | 0.941 | 0.828 | 0.972 | 0.948 | 0.899 | 0.761 | 0.584 | 0.859 | 0.593 |
| **GMM** 1    | 0.287 | 0.749 | 0.685 | 0.331 | 0.287 | 0.712 | 0.608 | 0.522 | 0.529 | 0.333 | 0.507 | 0.489 | 0.428 | 0.538 | 0.547 | 0.498 | 0.507 | 0.480 | 0.510 |
| **GMM** 1\*  | 0.320 | 0.358 | 0.916 | 0.314 | 0.672 | 0.302 | 0.366 | 0.486 | 0.538 | 0.324 | 0.672 | 0.152 | 0.058 | 0.568 | 0.584 | 0.360 | 0.532 | 0.492 | 0.522 |
| **GMM** 2    | 0.393 | 0.726 | 0.331 | 0.270 | 0.729 | 0.270 | 0.360 | 0.495 | 0.470 | 0.394 | 0.504 | 0.492 | 0.535 | 0.535 | 0.464 | 0.449 | 0.495 | 0.483 | 0.507 |
| **GMM** 2\*  | 0.626 | 0.342 | 0.751 | 0.430 | 0.707 | 0.691 | 0.633 | 0.428 | 0.510 | 0.327 | 0.663 | 0.336 | 0.593 | 0.477 | 0.486 | 0.513 | 0.425 | 0.480 | 0.483 |
| **GMM** 3    | 0.597 | 0.137 | 0.769 | 0.628 | 0.720 | 0.296 | 0.559 | 0.467 | 0.718 | 0.522 | 0.581 | 0.443 | 0.504 | 0.483 | 0.513 | 0.529 | 0.541 | 0.489 | 0.498 |
| **GMM** 3\*  | 0.684 | 0.262 | 0.327 | 0.689 | 0.335 | 0.709 | 0.648 | 0.547 | 0.501 | 0.308 | 0.709 | 0.131 | 0.152 | 0.834 | 0.434 | 0.660 | 0.538 | 0.431 | 0.525 |
| **LDA** 1    | 0.889 | 0.981 | 0.991 | 0.859 | 0.917 | 0.741 | 0.892 | 0.929 | 0.914 | 0.908 | 0.868 | 0.883 | 0.969 | 0.941 | 0.844 | 0.779 | 0.773 | 0.951 | 0.675 |
| **LDA** 1\*  | 0.947 | 0.986 | 0.999 | 0.885 | 0.923 | 0.773 | 0.966 | 0.966 | 0.987 | 0.975 | 0.966 | 0.850 | 0.990 | 0.987 | 0.914 | 0.862 | 0.828 | 0.951 | 0.721 |
| **LDA** 2    | 0.891 | 0.982 | 0.993 | 0.864 | 0.912 | 0.778 | 0.926 | 0.923 | 0.963 | 0.892 | 0.868 | 0.926 | 0.981 | 0.782 | 0.767 | 0.807 | 0.779 | 0.938 | 0.675 |
| **LDA** 2\*  | 0.907 | 0.983 | 0.996 | 0.891 | 0.930 | 0.795 | 0.960 | 0.926 | 0.993 | 0.960 | 0.957 | 0.960 | 0.981 | 0.938 | 0.908 | 0.856 | 0.788 | 0.938 | 0.666 |
| **LDA** 3    | 0.930 | 0.975 | 0.991 | 0.897 | 0.922 | 0.800 | 0.850 | 0.957 | 0.981 | 0.954 | 0.920 | 0.951 | 0.984 | 0.954 | 0.874 | 0.807 | 0.788 | 0.932 | 0.727 |
| **LDA** 3\*  | 0.964 | 0.982 | 0.998 | 0.922 | 0.938 | 0.805 | 0.984 | 0.972 | 0.978 | 0.975 | 0.984 | 0.966 | 0.993 | 0.981 | 0.941 | 0.914 | 0.819 | 0.948 | 0.773 |
| **SVC1** 1   | 0.712 | 0.712 | 0.712 | 0.712 | 0.712 | 0.712 | 0.474 | 0.522 | 0.529 | 0.492 | 0.507 | 0.489 | 0.498 | 0.538 | 0.492 | 0.504 | 0.507 | 0.510 | 0.516 |
| **SVC1** 1\* | 0.706 | 0.706 | 0.706 | 0.706 | 0.706 | 0.706 | 0.483 | 0.507 | 0.538 | 0.501 | 0.516 | 0.532 | 0.492 | 0.431 | 0.538 | 0.449 | 0.495 | 0.498 | 0.501 |
| **SVC1** 2   | 0.729 | 0.729 | 0.729 | 0.729 | 0.729 | 0.729 | 0.486 | 0.477 | 0.470 | 0.498 | 0.504 | 0.492 | 0.535 | 0.535 | 0.483 | 0.519 | 0.495 | 0.474 | 0.507 |
| **SVC1** 2\* | 0.705 | 0.705 | 0.705 | 0.705 | 0.705 | 0.705 | 0.519 | 0.458 | 0.510 | 0.477 | 0.519 | 0.458 | 0.474 | 0.501 | 0.510 | 0.467 | 0.507 | 0.519 | 0.507 |
| **SVC1** 3   | 0.720 | 0.720 | 0.720 | 0.720 | 0.720 | 0.720 | 0.440 | 0.513 | 0.510 | 0.522 | 0.501 | 0.455 | 0.504 | 0.535 | 0.513 | 0.529 | 0.541 | 0.495 | 0.495 |
| **SVC1** 3\* | 0.724 | 0.724 | 0.724 | 0.724 | 0.724 | 0.724 | 0.483 | 0.486 | 0.501 | 0.532 | 0.480 | 0.489 | 0.510 | 0.504 | 0.535 | 0.516 | 0.519 | 0.492 | 0.461 |
| **SVC2** 1   | 0.864 | 0.992 | 0.894 | 0.827 | 0.932 | 0.714 | 0.740 | 0.847 | 0.880 | 0.819 | 0.773 | 0.798 | 0.926 | 0.859 | 0.758 | 0.700 | 0.733 | 0.859 | 0.593 |
| **SVC2** 1\* | 0.945 | 0.995 | 0.903 | 0.882 | 0.925 | 0.742 | 0.957 | 0.957 | 0.957 | 0.966 | 0.978 | 0.920 | 0.969 | 0.948 | 0.892 | 0.831 | 0.697 | 0.840 | 0.642 |
| **SVC2** 2   | 0.866 | 0.993 | 0.887 | 0.825 | 0.921 | 0.729 | 0.718 | 0.813 | 0.889 | 0.779 | 0.785 | 0.825 | 0.941 | 0.844 | 0.706 | 0.663 | 0.700 | 0.892 | 0.556 |
| **SVC2** 2\* | 0.895 | 0.996 | 0.883 | 0.876 | 0.930 | 0.720 | 0.923 | 0.889 | 0.972 | 0.917 | 0.954 | 0.926 | 0.960 | 0.905 | 0.850 | 0.801 | 0.712 | 0.877 | 0.645 |
| **SVC2** 3   | 0.851 | 0.997 | 0.926 | 0.843 | 0.943 | 0.720 | 0.764 | 0.828 | 0.883 | 0.792 | 0.798 | 0.850 | 0.935 | 0.844 | 0.740 | 0.691 | 0.709 | 0.810 | 0.522 |
| **SVC2** 3\* | 0.959 | 0.995 | 0.943 | 0.888 | 0.935 | 0.768 | 0.981 | 0.899 | 0.917 | 0.981 | 0.969 | 0.905 | 0.978 | 0.960 | 0.932 | 0.825 | 0.703 | 0.877 | 0.657 |
| **GPC** 1    | 0.287 | 0.287 | 0.287 | 0.287 | 0.287 | 0.287 | 0.474 | 0.522 | 0.529 | 0.492 | 0.507 | 0.489 | 0.498 | 0.538 | 0.492 | 0.504 | 0.507 | 0.510 | 0.516 |
| **GPC** 1\*  | 0.293 | 0.293 | 0.293 | 0.293 | 0.293 | 0.293 | 0.483 | 0.507 | 0.538 | 0.501 | 0.516 | 0.532 | 0.492 | 0.568 | 0.538 | 0.550 | 0.495 | 0.498 | 0.501 |
| **GPC** 2    | 0.270 | 0.270 | 0.270 | 0.270 | 0.270 | 0.270 | 0.486 | 0.480 | 0.470 | 0.498 | 0.504 | 0.492 | 0.535 | 0.535 | 0.483 | 0.519 | 0.495 | 0.474 | 0.507 |
| **GPC** 2\*  | 0.294 | 0.294 | 0.294 | 0.294 | 0.294 | 0.294 | 0.519 | 0.541 | 0.510 | 0.477 | 0.519 | 0.541 | 0.474 | 0.501 | 0.510 | 0.467 | 0.507 | 0.519 | 0.507 |
| **GPC** 3    | 0.279 | 0.279 | 0.279 | 0.279 | 0.279 | 0.279 | 0.559 | 0.516 | 0.510 | 0.522 | 0.501 | 0.455 | 0.504 | 0.535 | 0.513 | 0.529 | 0.541 | 0.495 | 0.495 |
| **GPC** 3\*  | 0.275 | 0.275 | 0.275 | 0.275 | 0.275 | 0.275 | 0.483 | 0.486 | 0.501 | 0.532 | 0.480 | 0.489 | 0.510 | 0.504 | 0.535 | 0.516 | 0.519 | 0.492 | 0.461 |
| **RFC** 1    | 0.858 | 0.991 | 0.972 | 0.724 | 0.942 | 0.736 | 0.730 | 0.944 | 0.908 | 0.871 | 0.874 | 0.905 | 0.972 | 0.892 | 0.770 | 0.788 | 0.779 | 0.896 | 0.666 |
| **RFC** 1\*  | 0.864 | 0.996 | 0.974 | 0.729 | 0.931 | 0.719 | 0.957 | 0.966 | 0.935 | 0.831 | 0.935 | 0.926 | 0.993 | 0.926 | 0.819 | 0.850 | 0.810 | 0.905 | 0.672 |
| **RFC** 2    | 0.844 | 0.946 | 0.966 | 0.746 | 0.867 | 0.731 | 0.862 | 0.868 | 0.917 | 0.804 | 0.868 | 0.914 | 0.938 | 0.926 | 0.801 | 0.785 | 0.773 | 0.926 | 0.651 |
| **RFC** 2\*  | 0.867 | 0.978 | 0.965 | 0.779 | 0.907 | 0.729 | 0.889 | 0.896 | 0.948 | 0.899 | 0.911 | 0.911 | 0.978 | 0.911 | 0.834 | 0.816 | 0.785 | 0.951 | 0.666 |
| **RFC** 3    | 0.842 | 0.992 | 0.972 | 0.738 | 0.961 | 0.735 | 0.865 | 0.911 | 0.944 | 0.856 | 0.874 | 0.886 | 0.954 | 0.899 | 0.844 | 0.810 | 0.810 | 0.908 | 0.642 |
| **RFC** 3\*  | 0.908 | 0.997 | 0.939 | 0.825 | 0.947 | 0.764 | 0.880 | 0.905 | 0.920 | 0.886 | 0.825 | 0.911 | 0.975 | 0.911 | 0.926 | 0.788 | 0.798 | 0.929 | 0.706 |
| **MLP** 1    | 0.719 | 0.964 | 0.892 | 0.554 | 0.903 | 0.716 | 0.529 | 0.587 | 0.761 | 0.544 | 0.571 | 0.645 | 0.642 | 0.486 | 0.492 | 0.522 | 0.519 | 0.730 | 0.501 |
| **MLP** 1\*  | 0.922 | 0.519 | 0.874 | 0.796 | 0.723 | 0.646 | 0.935 | 0.886 | 0.712 | 0.944 | 0.951 | 0.785 | 0.951 | 0.908 | 0.862 | 0.712 | 0.541 | 0.385 | 0.507 |
| **MLP** 2    | 0.731 | 0.982 | 0.886 | 0.762 | 0.807 | 0.270 | 0.513 | 0.568 | 0.654 | 0.538 | 0.510 | 0.712 | 0.553 | 0.461 | 0.516 | 0.529 | 0.565 | 0.773 | 0.577 |
| **MLP** 2\*  | 0.812 | 0.962 | 0.923 | 0.812 | 0.707 | 0.635 | 0.914 | 0.819 | 0.899 | 0.905 | 0.926 | 0.853 | 0.948 | 0.825 | 0.822 | 0.382 | 0.519 | 0.865 | 0.620 |
| **MLP** 3    | 0.455 | 0.835 | 0.918 | 0.719 | 0.907 | 0.719 | 0.440 | 0.486 | 0.770 | 0.605 | 0.581 | 0.712 | 0.498 | 0.688 | 0.513 | 0.470 | 0.553 | 0.782 | 0.489 |
| **MLP** 3\*  | 0.916 | 0.727 | 0.934 | 0.767 | 0.912 | 0.730 | 0.920 | 0.798 | 0.767 | 0.938 | 0.941 | 0.669 | 0.186 | 0.926 | 0.865 | 0.804 | 0.522 | 0.834 | 0.535 |
| **ADC** 1    | 0.947 | 0.999 | 1.0   | 0.894 | 0.958 | 0.820 | 0.886 | 0.966 | 0.978 | 0.892 | 0.908 | 0.951 | 0.996 | 0.941 | 0.837 | 0.828 | 0.850 | 0.941 | 0.651 |
| **ADC** 1\*  | 0.982 | 0.997 | 1.0   | 0.916 | 0.960 | 0.835 | 0.978 | 0.978 | 0.993 | 0.975 | 0.978 | 0.963 | 0.993 | 0.978 | 0.917 | 0.868 | 0.896 | 0.957 | 0.746 |
| **ADC** 2    | 0.952 | 1.0   | 1.0   | 0.892 | 0.956 | 0.819 | 0.902 | 0.963 | 0.987 | 0.908 | 0.908 | 0.941 | 0.969 | 0.929 | 0.834 | 0.819 | 0.868 | 0.932 | 0.660 |
| **ADC** 2\*  | 0.951 | 0.999 | 1.0   | 0.917 | 0.969 | 0.842 | 0.969 | 0.981 | 0.996 | 0.966 | 0.972 | 0.981 | 1.0   | 0.948 | 0.926 | 0.862 | 0.877 | 0.935 | 0.737 |
| **ADC** 3    | 0.960 | 0.999 | 1.0   | 0.911 | 0.965 | 0.827 | 0.917 | 0.963 | 0.963 | 0.929 | 0.911 | 0.954 | 0.990 | 0.981 | 0.896 | 0.859 | 0.807 | 0.944 | 0.730 |
| **ADC** 3\*  | 0.987 | 1.0   | 1.0   | 0.922 | 0.970 | 0.835 | 0.993 | 0.960 | 0.996 | 0.981 | 0.984 | 0.987 | 0.990 | 0.990 | 0.960 | 0.889 | 0.886 | 0.960 | 0.758 |
| **GNB** 1    | 0.853 | 0.885 | 0.865 | 0.620 | 0.886 | 0.722 | 0.675 | 0.850 | 0.822 | 0.694 | 0.697 | 0.850 | 0.905 | 0.847 | 0.746 | 0.584 | 0.608 | 0.889 | 0.599 |
| **GNB** 1\*  | 0.933 | 0.875 | 0.850 | 0.745 | 0.841 | 0.714 | 0.822 | 0.932 | 0.847 | 0.703 | 0.685 | 0.853 | 0.923 | 0.886 | 0.828 | 0.633 | 0.608 | 0.862 | 0.614 |
| **GNB** 2    | 0.855 | 0.978 | 0.807 | 0.541 | 0.914 | 0.731 | 0.712 | 0.871 | 0.840 | 0.645 | 0.703 | 0.831 | 0.941 | 0.847 | 0.697 | 0.584 | 0.737 | 0.899 | 0.590 |
| **GNB** 2\*  | 0.862 | 0.854 | 0.856 | 0.703 | 0.814 | 0.704 | 0.752 | 0.847 | 0.853 | 0.752 | 0.721 | 0.840 | 0.883 | 0.862 | 0.782 | 0.571 | 0.629 | 0.840 | 0.550 |
| **GNB** 3    | 0.834 | 0.963 | 0.746 | 0.597 | 0.886 | 0.718 | 0.776 | 0.844 | 0.825 | 0.694 | 0.678 | 0.819 | 0.926 | 0.868 | 0.556 | 0.605 | 0.605 | 0.868 | 0.556 |
| **GNB** 3\*  | 0.941 | 0.981 | 0.700 | 0.536 | 0.919 | 0.748 | 0.764 | 0.871 | 0.840 | 0.740 | 0.681 | 0.840 | 0.941 | 0.868 | 0.813 | 0.642 | 0.614 | 0.822 | 0.657 |
| **QDA** 1    | 0.316 | 0.762 | 0.956 | 0.600 | 0.755 | 0.656 | 0.755 | 0.645 | 0.859 | 0.678 | 0.758 | 0.874 | 0.984 | 0.828 | 0.721 | 0.633 | 0.617 | 0.850 | 0.568 |
| **QDA** 1\*  | 0.524 | 0.920 | 0.912 | 0.820 | 0.777 | 0.713 | 0.917 | 0.883 | 0.957 | 0.948 | 0.874 | 0.871 | 0.990 | 0.948 | 0.883 | 0.801 | 0.516 | 0.862 | 0.733 |
| **QDA** 2    | 0.765 | 0.930 | 0.884 | 0.514 | 0.721 | 0.573 | 0.850 | 0.840 | 0.840 | 0.691 | 0.740 | 0.862 | 0.966 | 0.810 | 0.688 | 0.623 | 0.764 | 0.874 | 0.556 |
| **QDA** 2\*  | 0.795 | 0.965 | 0.825 | 0.527 | 0.799 | 0.586 | 0.782 | 0.951 | 0.865 | 0.782 | 0.761 | 0.886 | 0.990 | 0.929 | 0.779 | 0.605 | 0.516 | 0.911 | 0.758 |
| **QDA** 3    | 0.772 | 0.968 | 0.893 | 0.499 | 0.678 | 0.521 | 0.547 | 0.853 | 0.871 | 0.813 | 0.727 | 0.859 | 0.990 | 0.889 | 0.672 | 0.617 | 0.660 | 0.856 | 0.602 |
| **QDA** 3\*  | 0.949 | 0.967 | 0.998 | 0.443 | 0.851 | 0.563 | 0.935 | 0.905 | 0.911 | 0.935 | 0.975 | 0.920 | 0.981 | 0.929 | 0.730 | 0.681 | 0.529 | 0.828 | 0.672 |
| **NB** 1     | 0.733 | 0.713 | 0.876 | 0.711 | 0.711 | 0.712 | 0.703 | 0.770 | 0.645 | 0.596 | 0.654 | 0.611 | 0.568 | 0.642 | 0.599 | 0.474 | 0.525 | 0.663 | 0.467 |
| **NB** 1\*   | 0.744 | 0.706 | 0.870 | 0.706 | 0.706 | 0.706 | 0.669 | 0.743 | 0.672 | 0.568 | 0.623 | 0.574 | 0.568 | 0.657 | 0.626 | 0.507 | 0.516 | 0.651 | 0.492 |
| **NB** 2     | 0.733 | 0.719 | 0.810 | 0.728 | 0.726 | 0.729 | 0.678 | 0.755 | 0.642 | 0.584 | 0.629 | 0.593 | 0.565 | 0.617 | 0.587 | 0.474 | 0.522 | 0.654 | 0.495 |
| **NB** 2\*   | 0.707 | 0.703 | 0.777 | 0.705 | 0.705 | 0.705 | 0.694 | 0.724 | 0.663 | 0.596 | 0.623 | 0.620 | 0.544 | 0.660 | 0.611 | 0.507 | 0.522 | 0.648 | 0.461 |
| **NB** 3     | 0.720 | 0.729 | 0.852 | 0.720 | 0.717 | 0.720 | 0.672 | 0.764 | 0.645 | 0.568 | 0.642 | 0.657 | 0.538 | 0.633 | 0.617 | 0.513 | 0.522 | 0.648 | 0.519 |
| **NB** 3\*   | 0.732 | 0.742 | 0.868 | 0.724 | 0.723 | 0.724 | 0.700 | 0.779 | 0.639 | 0.562 | 0.648 | 0.571 | 0.544 | 0.629 | 0.565 | 0.498 | 0.538 | 0.672 | 0.495 |

![A01](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A01.png)
![A02](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A02.png)
![A03](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A03.png)
![A04](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A04.png)
![A05](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A05.png)
![A06](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A06.png)
![A07](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A07.png)
![A08](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A08.png)
![A09](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A09.png)
![A10](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A10.png)
![A11](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A11.png)
![A12](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A12.png)
![A13](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A13.png)
![A14](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A14.png)
![A15](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A15.png)
![A16](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A16.png)
![A17](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A17.png)
![A18](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A18.png)
![A19](https://raw.githubusercontent.com/Helias/audio-stats/main/system_ids_accuracies/A19.png)

# old experiment results

A07, **10284 samples**, 48 speakers

| Model    | Accuracy |
| -------- | -------- |
| **CART** | 0.9      |

---

A07, **964 samples**, 5 speakers

| Model    | Accuracy |
| -------- | -------- |
| **CART** | 0.938    |
| **SVM**  | 0.941    |
| **LR**   | 0.483    |
| **KNN**  | 0.944    |
| **GMM**  | 0.079    |
| **LDA**  | 0.948    |
| **SVC1** | 0.522    |
| **SVC2** | 0.948    |
| **GPC**  | 0.944    |
| **RFC**  | 0.944    |
| **MLP**  | 0.483    |
| **ADC**  | 0.944    |
| **GNB**  | 0.944    |
| **QDA**  | 0.944    |
| **NB**   | 0.516    |

---

Average accuracy generated using 10 iterations classifying audio with synthetic techniques where the bit rate is discriminant.

System IDs: A07, A10, A11, A13, A14  
Speakers involved (5): LA_0012, LA_0013, LA_0047, LA_0023, LA_0038

| model    | A07   | A10   | A11   | A13   | A14   |
| -------- | ----- | ----- | ----- | ----- | ----- |
| **CART** | 0.938 | 0.954 | 0.905 | 0.951 | 0.859 |
| **SVM**  | 0.957 | 0.948 | 0.929 | 0.969 | 0.880 |
| **LR**   | 0.510 | 0.510 | 0.470 | 0.452 | 0.489 |
| **KNN**  | 0.963 | 0.948 | 0.923 | 0.966 | 0.880 |
| **GMM**  | 0.957 | 0.957 | 0.064 | 0.033 | 0.837 |
| **LDA**  | 0.954 | 0.941 | 0.932 | 0.963 | 0.877 |
| **SVC1** | 0.495 | 0.495 | 0.532 | 0.461 | 0.529 |
| **SVC2** | 0.957 | 0.957 | 0.926 | 0.960 | 0.883 |
| **GPC**  | 0.957 | 0.948 | 0.932 | 0.966 | 0.880 |
| **RFC**  | 0.948 | 0.960 | 0.914 | 0.951 | 0.880 |
| **MLP**  | 0.510 | 0.510 | 0.529 | 0.452 | 0.489 |
| **ADC**  | 0.954 | 0.954 | 0.911 | 0.948 | 0.880 |
| **GNB**  | 0.960 | 0.948 | 0.932 | 0.966 | 0.883 |
| **QDA**  | 0.960 | 0.948 | 0.932 | 0.966 | 0.883 |
| **NB**   | 0.489 | 0.489 | 0.529 | 0.452 | 0.510 |

---

Average accuracy generated using 10 iterations classifying audio with synthetic techniques where the bit rate is not discriminant.

**System IDs**: A08, A09, A12, A15, A16, A17, A18, A19  
**Speakers involved (5)**: LA_0012, LA_0013, LA_0047, LA_0023, LA_0038

| model    | A08   | A09   | A12   | A15   | A16   | A17   | A18   | A19   |
| -------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| **CART** | 0.792 | 0.770 | 0.767 | 0.856 | 0.724 | 0.532 | 0.547 | 0.535 |
| **SVM**  | 0.856 | 0.840 | 0.813 | 0.902 | 0.755 | 0.470 | 0.571 | 0.495 |
| **LR**   | 0.501 | 0.486 | 0.477 | 0.474 | 0.477 | 0.477 | 0.440 | 0.498 |
| **KNN**  | 0.850 | 0.807 | 0.810 | 0.886 | 0.740 | 0.525 | 0.623 | 0.553 |
| **GMM**  | 0.847 | 0.293 | 0.330 | 0.892 | 0.327 | 0.544 | 0.422 | 0.415 |
| **LDA**  | 0.868 | 0.850 | 0.807 | 0.892 | 0.749 | 0.477 | 0.645 | 0.553 |
| **SVC1** | 0.501 | 0.519 | 0.535 | 0.544 | 0.532 | 0.477 | 0.452 | 0.501 |
| **SVC2** | 0.862 | 0.844 | 0.813 | 0.902 | 0.749 | 0.477 | 0.611 | 0.544 |
| **GPC**  | 0.862 | 0.847 | 0.822 | 0.905 | 0.752 | 0.535 | 0.568 | 0.544 |
| **RFC**  | 0.853 | 0.807 | 0.816 | 0.892 | 0.733 | 0.510 | 0.620 | 0.544 |
| **MLP**  | 0.501 | 0.513 | 0.477 | 0.474 | 0.477 | 0.522 | 0.559 | 0.498 |
| **ADC**  | 0.837 | 0.822 | 0.816 | 0.902 | 0.749 | 0.535 | 0.605 | 0.550 |
| **GNB**  | 0.865 | 0.847 | 0.795 | 0.899 | 0.749 | 0.477 | 0.602 | 0.559 |
| **QDA**  | 0.865 | 0.847 | 0.795 | 0.899 | 0.749 | 0.477 | 0.602 | 0.559 |
| **NB**   | 0.498 | 0.513 | 0.522 | 0.525 | 0.522 | 0.477 | 0.440 | 0.498 |

---

Average accuracy generated using 10 iterations classifying audio with synthetic techniques where the bit rate is not discriminant. (part 2)

**System IDs**: A01, A02, A03, A04, A05, A06  
**Speakers involved (7)**: LA_0069, LA_0070, LA_0071, LA_0072, LA_0073, LA_0074, LA_0075

| model    | A01   | A02   | A03   | A04   | A05   | A06   |
| -------- | ----- | ----- | ----- | ----- | ----- | ----- |
| **CART** | 0.872 | 0.585 | 0.628 | 0.755 | 0.611 | 0.645 |
| **SVM**  | 0.919 | 0.542 | 0.499 | 0.833 | 0.510 | 0.466 |
| **LR**   | 0.711 | 0.711 | 0.692 | 0.713 | 0.711 | 0.711 |
| **KNN**  | 0.915 | 0.683 | 0.694 | 0.820 | 0.691 | 0.718 |
| **GMM**  | 0.084 | 0.489 | 0.700 | 0.172 | 0.493 | 0.342 |
| **LDA**  | 0.917 | 0.711 | 0.726 | 0.833 | 0.711 | 0.714 |
| **SVC1** | 0.710 | 0.707 | 0.689 | 0.711 | 0.706 | 0.705 |
| **SVC2** | 0.913 | 0.711 | 0.720 | 0.832 | 0.711 | 0.723 |
| **GPC**  | 0.914 | 0.711 | 0.729 | 0.833 | 0.711 | 0.726 |
| **RFC**  | 0.912 | 0.711 | 0.712 | 0.829 | 0.706 | 0.720 |
| **MLP**  | 0.711 | 0.711 | 0.692 | 0.286 | 0.711 | 0.711 |
| **ADC**  | 0.913 | 0.707 | 0.716 | 0.820 | 0.710 | 0.721 |
| **GNB**  | 0.914 | 0.711 | 0.725 | 0.833 | 0.711 | 0.712 |
| **QDA**  | 0.914 | 0.711 | 0.725 | 0.833 | 0.711 | 0.712 |
| **NB**   | 0.711 | 0.711 | 0.692 | 0.713 | 0.711 | 0.711 |
