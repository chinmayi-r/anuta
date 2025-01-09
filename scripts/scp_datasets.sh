#!/usr/bin/env bash

scp data/cidds*all.csv "$1":~/anuta/data/
scp data/syn/* "$1":~/anuta/data/syn/