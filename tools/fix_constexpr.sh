#!/bin/bash
perl -0777 -i.original -pe 's/([^\n]*if[^\n]*)\n[ ]*(constexpr[^\n]*\n)/\1 \2/gmis' $1
