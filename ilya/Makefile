DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

github: 
	git add ${DIR}/..
	git commit -m "Default msg"
	git push