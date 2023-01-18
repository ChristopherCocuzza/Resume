soft = dssv.f

dssvlib:
	rm -f *.so
	f2py -c \
	-m dssvlib \
	--fcompiler=gnu95 \
	--f90flags="-Wtabs" \
	--f77flags="-std=legacy" \
	$(soft) 

clean:
	rm *.so  pro

#-ffixed-line-length-none 
#--f77flags="-std=legacy" \




