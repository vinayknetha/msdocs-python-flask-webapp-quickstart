const form = document.querySelector('form');
const statusMessage = document.getElementById('statusMessage');
const submitButton = document.querySelector('button');
const fileInput = document.querySelector('input');
const progressBar = document.querySelector('progress');
const fileNum = document.getElementById('fileNum');
const fileListMetadata = document.getElementById('fileListMetadata');

form.addEventListener('submit', handleSubmit);
fileInput.addEventListener('change', handleInputChange);

function handleSubmit(event) {
  event.preventDefault();
  //alert("hi");
  showPendingState();

  uploadFiles();
}

function handleInputChange() {
  resetFormState();

  try {
    assertFilesValid(fileInput.files);
  } catch (err) {
    updateStatusMessage(err.message);
    return;
  }

  submitButton.disabled = false;
}

function uploadFiles() {
  const url = 'http://127.0.0.1:5000/upload';
  const method = 'post';

  const xhr = new XMLHttpRequest();

  //const data = new FormData(form);



  /*var data = new FormData();
  data.append("file", fileInput.files[0], form);
   
  var xhr = new XMLHttpRequest();
  xhr.withCredentials = true;
  
  xhr.addEventListener("readystatechange", function() {
    if(this.readyState === 4) {
      console.log(this.responseText);
    }
  });
  
  xhr.open("POST", "http://127.0.0.1:5000/upload");
  
  xhr.send(data);
*/
  const formdata = new FormData();
formdata.append("file", fileInput.files[0], form);


const requestOptions = {
  method: "POST",
  body: formdata,
  redirect: "follow"
};

fetch("http://127.0.0.1:5000/upload", requestOptions)
  .then((response) => response.text())
  .then((result) => {console.log(result)
  document.getElementById("writeHere").innerHTML=JSON.parse(result).AreasHTML;
  }

)
  .catch((error) => console.error(error));



  xhr.upload.addEventListener('progress', (event) => {
    updateStatusMessage(`⏳ Uploaded ${event.loaded} bytes of ${event.total}`);
    updateProgressBar(event.loaded / event.total);
  });

  xhr.addEventListener('loadend', () => {
    if (xhr.status === 200) {
      updateStatusMessage('✅ Success');
      renderFilesMetadata(fileInput.files);
    } else {
      updateStatusMessage('❌ Error');
    }

    updateProgressBar(0);
  });

  xhr.open(method, url);
  xhr.send(data);
}

function updateStatusMessage(text) {
  statusMessage.textContent = text;
}

function showPendingState() {
  submitButton.disabled = true;
  updateStatusMessage('⏳ Pending...');
}

function resetFormState() {
  submitButton.disabled = true;
  updateStatusMessage(`🤷‍♂ Nothing's uploaded`);

  fileListMetadata.textContent = '';
  fileNum.textContent = '0';
}

function updateProgressBar(value) {
  const percent = value * 100;
  progressBar.value = Math.round(percent);
}

function renderFilesMetadata(fileList) {
  fileNum.textContent = fileList.length;

  fileListMetadata.textContent = '';

  for (const file of fileList) {
    const name = file.name;
    const type = file.type;
    const size = file.size;

    fileListMetadata.insertAdjacentHTML(
      'beforeend',
      `
        <li>
          <p><strong>Name:</strong> ${name}</p>
          <p><strong>Type:</strong> ${type}</p>
          <p><strong>Size:</strong> ${size} bytes</p>
        </li>`,
    );
  }
}

function assertFilesValid(fileList) {
  const allowedTypes = ['image/webp', 'image/jpeg', 'image/png'];
  const sizeLimit = 1024 * 1024 * 1024; // 1 GB

  for (const file of fileList) {
    const { name: fileName, size: fileSize } = file;

    if (!allowedTypes.includes(file.type)) {
      throw new Error(
        `❌ File "${fileName}" could not be uploaded. Only images with the following types are allowed: WEBP, JPEG, PNG.`,
      );
    }

    if (fileSize > sizeLimit) {
      throw new Error(
        `❌ File "${fileName}" could not be uploaded. Only images up to 1 MB are allowed.`,
      );
    }
  }
}
