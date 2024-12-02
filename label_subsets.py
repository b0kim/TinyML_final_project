import requests

IMAGENET_LABELS = requests.get("https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json").json()

FURNITURE_LABELS = ["barber chair",
					"bookcase",
					"china cabinet",
					"chiffonier",
					"chest",
					"cradle",
					"desk",
					"dining table",
					"filing cabinet",
					"folding chair",
					"four-poster bed",
					"infant bed",
					"medicine chest",
					"rocking chair",
					"sofa",
					"wardrobe"]
