#!/bin/bash

program=$1
#kill -9 `ps aux | grep "$grogram" | grep -v grep |awk '{print $2}'` > /dev/null 2>&1
echo "kill $program"
kill -9 `ps aux | grep "$grogram" | grep -v grep |awk '{print $2}'`
