import pytest
import requests
import json


@pytest.mark.parametrize("userid, firstname", [(1, "George"), (2, "Janet"), (3, "Emma")])
def test_list_valid_user(supply_url, userid, firstname):
	url = supply_url + "/users/" + str(userid)
	resp = requests.get(url)
	j = json.loads(resp.text)
	assert resp.status_code == 200, resp.text
	assert j['data']['id'] == userid, resp.text
	assert j['data']['first_name'] == firstname, resp.text


@pytest.mark.parametrize("userid", [23, 50])
def test_list_invaliduser(supply_url, userid):
	url = supply_url + "/users/"+ str(userid)
	resp = requests.get(url)
	assert resp.status_code == 404, resp.tex
