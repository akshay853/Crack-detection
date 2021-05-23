document.querySelector(".btn").addEventListener("submit",submitForm);

function submitForm(e)
{
    e.preventDefault();
    let name = document.querySelector(".name").value();
    console.log(name);
}