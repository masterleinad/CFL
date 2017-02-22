#!/bin/bash
perl -0777 -i.original -pe 's/[ ]*\{[ ]*\n([^\n]*)\n[ ]*\}[ ]*/\1/gmis' $1
