import pytest


@pytest.mark.set1
def test_file1_method1():
	x = 5
	y = 6
	assert x + 1 == y, 'some text 2'
	assert x == y, 'test failed because x=' + str(x) + ' y=' + str(y)
	assert x == y, 'some text 1'
	assert 1 == 2, 'one is two'

# In Python Pytest, if an assertion fails in a test method, then that 
# method execution is stopped there. The remaining code in that test 
# method is not executed, and Pytest assertions will continue with the next test method.
@pytest.mark.set2
def test_file1_method2():
	x = 5
	y = 6
	assert x + 1 == y, 'some sext 3'
