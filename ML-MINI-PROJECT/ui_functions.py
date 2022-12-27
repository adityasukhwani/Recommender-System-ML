from call_api import call,call_genre
# from main import recommend

def display(userID):
    data = {
        "userId" : userID,
        "moviename": "string",
    }
    movies = call(data)
    return movies

def display_genre(movieName):
    data={
        "userId" : 0,
        "moviename" : movieName,
    }
    movies=call_genre(data)
    return movies