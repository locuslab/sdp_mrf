git clone https://github.com/lucasb-eyer/pydensecrf.git
cp vectorInf.diff pydensecrf/
cd pydensecrf/
git apply vectorInf.diff
python setup.py install
cd ..

git clone https://github.com/sejunpark-repository/spectral_approximate_inference.git
cp check.m spectral_approximate_inference/matlab_code