import requests
import json

def main():
    # print(tracks(6))
    return tracks(6)

def tracks(n):
    response = requests.get("https://itunes.apple.com/search?entity=song&limit=" + str(n) + "&term=two_steps_from_hell")
    return [result["trackName"] for result in response.json()["results"]]

if __name__ == "__main__":
    print(main())
