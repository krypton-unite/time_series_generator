python setup.py sdist bdist_wheel
$version="0.2.6"
$files_to_handle_str="dist/time_series_generator-$version*" 
twine check $files_to_handle_str
twine upload $files_to_handle_str