# Set Up Google Cloud for this Project

This project uses <a href="https://cloud.google.com/translate/">Google Cloud Translate</a> to translate the English questions of LC-QuAD 2.0 into Dutch. Thus, if you want to fully reproduce this project's dataset creation, you need to set up this translation service for yourself as well.

You need to take six steps in order to enable Google Cloud Translation for this project:

1. Create a Google Cloud account.
2. Create a project.
3. Enable the 'Cloud Translation API'.
4. Create a service account for the 'Cloud Translation API'.
5. Create a service account key for the service account.
6. Enable billing for the project.

Each step is explained more thoroughly below.

**Note.** Please realise that these instructions may not apply anymore due changes made by Google. This guide applies to Google Cloud as it stands on the 28th of July, 2022. Refer to the documentation of <a href="https://support.google.com/googlecloud/cloud/?hl=en">Google Cloud</a> for up-to-date information. Specifically, refer to the two articles <a href="https://cloud.google.com/translate/docs/setup?hl=en_US">Cloud Translation: Setup</a> and <a href="https://cloud.google.com/translate/docs/basic/translate-text-basic?hl=en_US">Cloud Translation: Translate text (Basic edition)</a> if these are still available.

## Step 1: Create a Google Cloud account

Visit the <a href="https://cloud.google.com/">Google Cloud homepage</a> and click 'Get started for free'. Follow the steps to create an account. Note that you need to provide credit card information, even though by default automatic billing is inactive.

## Step 2: Create a project

Once you have created an account, create a project by following these steps:

1. Click the 'hamburger menu' at the top-left of the page. It's the three horizontally stacked lines.
2. Hover over (but _don't_ click on) 'IAM & Admin'.
3. Click on 'Manage Resources'.
4. Click 'Create project'.
5. Give the project a short, descriptive name, such as 'Dutch KBQA'.
6. Choose a location. For individual accounts, use 'No organization'. For organisational accounts, choose the name of the organisation.
7. Click 'Create'.

## Step 3: Enable the 'Cloud Translation API'

Next, you need to enable Google's Cloud Translation API for your just-created project. Here is how it is done:

1. Click the 'hamburger menu' at the top-left of the page.
2. Click 'View all products'.
3. Click 'Visit marketplace'.
4. Search for 'Cloud Translation API' using the 'Search Marketplace' search box at the top of the page.
5. Select 'Cloud Translation API' from the list of results. (The owner of the product is 'Google Enterprise API'.) 
6. Click 'Enable'.
7. Wait a couple of seconds for the API to get enabled.

**Note 1.** You may need to select your project before enabling the API in step 3.6. You can this by clicking the drop-down directly to the right of the 'hamburger menu' and selecting your project.

**Note 2.** Don't confuse 'Cloud Translation API' with 'Translation'. The former product enables you to translate text using Google's pre-trained language models, as is done on <a href="https://translate.google.com/">Google Translate</a>. The latter product allows you to "[t]rain a custom model using your own dataset of sentence pairs", instead. It's also known as the 'AutoML API'.

## Step 4: Create a service account for the 'Cloud Translation API'

On the screen that you just landed on, perform the following steps:

1. Click 'Create credentials' near the top-right corner.
2. Under the header 'What data will you be accessing?', select 'Application data'.
3. You're presented with the question 'Are you planning to use this API with Compute Engine, Kubernetes Engine, App Engine, or Cloud Functions?'. Select 'No, I'm not using them'.
4. Click 'Next'.
5. Under 'Service account details', fill in at least the following two fields:
	1. The service account name. Example: 'Dutch KBQA Service Account'.
	2. The service account ID. Example: 'dutch-kbqa-service-account'.
6. Click 'Create and continue'.
7. Select the role 'Cloud Translation API User'.
8. Click 'Continue'.
9. Click 'Done'.

**Context.** According to <a href="https://cloud.google.com/iam/docs/service-accounts">this Google Cloud help page</a>, a 'service account' is "[...] a special kind of account used by an [app], rather than a person." In short, service accounts allow apps to authorise themselves to Google Cloud. Such authorisation is needed to determine whether the app has 'a right' to access Google Cloud's APIs. Apps may not have this 'right' for various reasons: they may not possess enough funds, they may have been banned from Google Cloud, and so on.

## Step 5: Create a service account key for the service account

After the creation of the 'Cloud Translation API' service account, you should have returned to the 'API/Service Details' page of the 'Cloud Translation API'. Now perform the following actions:

1. Under the header 'Service Accounts', click the ID of your just-created service account.
2. Click 'Keys'.
3. Click 'Add key' > 'Create new key'.
4. Click 'Create'. A service account key in JSON format is downloaded to your computer.
5. Click close.
6. Move the downloaded key. In a terminal, run `mv ~/Downloads/$GC_KEY.json .service-account-key.json`, where `$GC_KEY` represents the name of the JSON file downloaded in step 5.4.

## Step 6: Enable billing for the project

Finally, you need to enable billing on the project. Realise that translating the LC-QuAD 2.0 dataset into a Dutch version costs money; you use Google Cloud's APIs. Hence, it may be necessary to top up the primary payment method used for the project. Also see the note below.

To enable billing for the project:

1. Go to the Google Cloud Console homepage by following <a href="https://console.cloud.google.com">this link</a>, or by clicking the Google Cloud logo directly to the right of the 'hamburger menu'. (The Google Cloud Console homepage _is not_ the Google Cloud homepage! The former is for users of Google Cloud; the latter is the 'front page' for Google Cloud, for users and non-users alike.)
2. Make sure your just-created project is chosen by selecting your project in the dropdown directly to the right of the Google Cloud logo.
3. Click the 'hamburger menu' at the top-left of the page.
4. Click 'Billing'.
5. If a warning or error is shown when going to the page, follow the steps listed <a href="https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#billing_project_linkage">here</a>. If nothing extraordinary happens, continue to step 6.6.
6. Click 'Overview' if you're not already there.
7. Click 'Payment overview'.
8. If your payment method has any problems, there are prominently shown on the page you just landed on. If you see nothing extraordinary, this step is successful. If, instead, you _do_ see one or more prominent warnings or errors, follow the steps listed <a href="https://cloud.google.com/billing/docs/how-to/resolve-issues#resolving_declined_payments">here</a>.

**Note 1.** As of the 28th of June, 2022, all new users of Google Cloud receive $300 worth of service on the platform; this is (more than) sufficient for translating the dataset.

**Note 2.** If you encounter any problem during step 6, read <a href="https://cloud.google.com/billing/docs/how-to/verify-billing-enabled">this Google Cloud help center article</a> thoroughly.

