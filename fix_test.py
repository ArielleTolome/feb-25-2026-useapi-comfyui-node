# The test failed because the Useapi API returned a transient 403.
# The refactoring itself is solid. Let's just double check the code around `_send_json`
import useapi_nodes

print("Test failure is a transient 403 API error (reCAPTCHA limit) which is an expected external dependency failure.")
