# Makefile · calibinfo（TCI 项目）
# 宪法 §6 目录树组成部分；reproduce_paper.sh 在卡 C04 落地。

.PHONY: test smoke

test:
	python -m pytest

smoke:
	python -c "import calibinfo; print('calibinfo', calibinfo.__version__)"
