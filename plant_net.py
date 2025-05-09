import requests
import os
from typing import Dict, Any
import json
import tempfile

class PlantNetAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://my-api.plantnet.org/v2/identify/all"

    def identify_plant_from_path(self, image_path: str) -> Dict[str, Any]:
        """
        Identify plant from image path with robust error handling.
        
        Returns:
            Dict with either:
            - Success: {'scientific_name': str, 'common_name': str, 'confidence': float, 'raw_data': dict}
            - Error: {'error': str}
        """
        try:
            if not self.api_key or self.api_key == "your_plantnet_api_key_here":
                 return {'error': "PlantNet API Key is not configured."}
            # Validate image exists
            if not os.path.exists(image_path):
                return {'error': "Image file not found"}

            with open(image_path, 'rb') as img_file:
                files = [('images', (os.path.basename(image_path), img_file, 'image/jpeg'))] # Assuming jpeg, can be more dynamic
                params = {'api-key': self.api_key, 'include-related-images': 'false'}
                data = {'organs': ['auto']} # Specify organs for better results if known, 'auto' is general

                response = requests.post(
                    self.base_url,
                    files=files,
                    params=params,
                    data=data,
                    timeout=20 # Increased timeout
                )

                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('message', response.text)
                        # Provide more specific error for common issues
                        if response.status_code == 401: error_msg = "Unauthorized - Check PlantNet API Key."
                        if response.status_code == 404 and "No result" in error_msg : error_msg = "No plant matches found by API."
                        elif response.status_code == 400 : error_msg = f"Bad Request: {error_msg}"

                    except json.JSONDecodeError:
                        error_msg = response.text
                    return {'error': f"API Error {response.status_code}: {error_msg}"}

                return self._parse_response(response.json())

        except requests.exceptions.Timeout:
            return {'error': "Network/API Error: Request to PlantNet API timed out."}
        except requests.exceptions.RequestException as e:
            return {'error': f"Network/API Error: {str(e)}"}
        except json.JSONDecodeError as e:
            return {'error': f"API Response Error: Could not decode JSON from PlantNet. {str(e)}"}
        except Exception as e:
            return {'error': f"Unexpected Error during plant identification: {str(e)}"}

    def identify_plant_from_bytes(self, image_bytes: bytes, filename: str = "upload.jpg") -> Dict[str, Any]:
        """
        Identify plant from image bytes using a temporary file.
        """
        try:
            if not self.api_key or self.api_key == "your_plantnet_api_key_here":
                 return {'error': "PlantNet API Key is not configured."}
            # Use a temporary file to pass to identify_plant_from_path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_file_path = tmp_file.name
            
            result = self.identify_plant_from_path(tmp_file_path)
            
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        return result

    def _parse_response(self, api_response: Dict[str, Any]) -> Dict[str, Any]:
        """Safely parse the API response."""
        try:
            if not isinstance(api_response, dict):
                return {'error': "Invalid API response format: Expected a dictionary."}

            results = api_response.get('results') # results can be None or empty list
            if not results: # Handles None or empty list
                return {'error': "No plant matches found in API response."}
            
            # Check if results is a list and non-empty
            if not isinstance(results, list) or not results:
                 return {'error': "No plant matches found in API response (results empty or not a list)."}


            best_match = results[0]
            if not isinstance(best_match, dict):
                return {'error': "Invalid match data format: Expected a dictionary for best_match."}

            score = best_match.get('score')
            if score is None: # Could be 0.0, so check for None explicitly
                return {'error': "Match data missing 'score'."}

            species_data = best_match.get('species')
            if not isinstance(species_data, dict):
                return {'error': "Invalid species data format: Expected a dictionary."}

            scientific_name = species_data.get('scientificNameWithoutAuthor', 'Unknown Scientific Name')
            
            common_names = species_data.get('commonNames', [])
            if not isinstance(common_names, list): common_names = ['Unknown Common Name']
            common_name = (common_names[0] if common_names else 'Unknown Common Name')

            # Handle cases where common_names might be null or empty list from API
            if not common_name or common_name.strip() == "":
                common_name = 'Unknown Common Name'

            return {
                'scientific_name': scientific_name,
                'common_name': common_name,
                'confidence': round(float(score) * 100, 1) if score is not None else 0.0,
                'raw_data': api_response # Return the whole response for more detailed debugging if needed
            }

        except KeyError as e:
            return {'error': f"Response parsing error: Missing key {str(e)}."}
        except (TypeError, ValueError) as e: # Catches issues with float conversion or list indexing
            return {'error': f"Response parsing error: Data type or value issue - {str(e)}."}
        except Exception as e:
            return {'error': f"Unexpected response parsing error: {str(e)}"}