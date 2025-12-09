const fs = require("fs");
const path = require("path");

// Helper function to generate patient data
const generatePatients = () => {
  const firstNames = {
    male: [
      "Văn",
      "Hữu",
      "Minh",
      "Đức",
      "Tuấn",
      "Hoàng",
      "Quang",
      "Thành",
      "Duy",
      "Huy",
      "Tấn",
      "Công",
      "Trung",
      "Thanh",
      "Khải",
      "Bảo",
      "Long",
      "Nam",
      "Sơn",
      "Hải",
    ],
    female: [
      "Thị",
      "Kim",
      "Thu",
      "Lan",
      "Mai",
      "Hoa",
      "Hương",
      "Linh",
      "Ngọc",
      "Phương",
      "Như",
      "Ánh",
      "Thảo",
      "Trang",
      "Vy",
      "Quyên",
      "Yến",
      "Châu",
      "Nhi",
      "Anh",
    ],
  };

  const lastNames = [
    "Nguyễn",
    "Trần",
    "Lê",
    "Phạm",
    "Hoàng",
    "Huỳnh",
    "Phan",
    "Vũ",
    "Võ",
    "Đặng",
    "Bùi",
    "Đỗ",
    "Hồ",
    "Ngô",
    "Dương",
    "Lý",
    "Đinh",
    "Cao",
    "Trịnh",
    "Tô",
    "Lưu",
    "Mai",
    "Chu",
    "Lâm",
    "Đoàn",
  ];

  const middleNames = {
    male: [
      "An",
      "Bình",
      "Cường",
      "Dũng",
      "Em",
      "Giang",
      "Hùng",
      "Khánh",
      "Linh",
      "Mạnh",
      "Nhân",
      "Phúc",
      "Quân",
      "Sĩ",
      "Tài",
      "Vĩ",
      "Xuân",
      "Yên",
      "Anh",
      "Chí",
    ],
    female: [
      "Anh",
      "Bích",
      "Chi",
      "Diệu",
      "Em",
      "Giang",
      "Hà",
      "Kim",
      "Linh",
      "My",
      "Ngân",
      "Oanh",
      "Phương",
      "Quỳnh",
      "Thanh",
      "Uyên",
      "Vân",
      "Xuân",
      "Yến",
      "Dung",
    ],
  };

  const diagnoses = [
    {
      name: "Lao phổi",
      image: "/src/mock_data/patient_data/01_Tuberculosis/origin.png",
    },
    {
      name: "Viêm phổi",
      image: "/src/mock_data/patient_data/02_pneumonia/origin.png",
    },
    {
      name: "Bệnh phổi khác",
      image: "/src/mock_data/patient_data/03_Otherdisease/origin.png",
    },
    {
      name: "Chưa phát hiện",
      image: "/src/mock_data/patient_data/07_Nofinding/origin.png",
    },
    {
      name: "Lao phổi",
      image: "/src/mock_data/patient_data/04_Turbeculosis/origin.png",
    },
    {
      name: "Viêm phổi",
      image: "/src/mock_data/patient_data/05_pneumonia/origin.png",
    },
    {
      name: "Viêm phổi",
      image: "/src/mock_data/patient_data/06_pneumonia/origin.png",
    },
  ];

  const statuses = ["Improving", "Stable", "Critical"];
  const bloodTypes = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"];

  const patients = [];

  // Set seed for reproducible results
  let seed = 12345;
  const random = () => {
    const x = Math.sin(seed++) * 10000;
    return x - Math.floor(x);
  };

  for (let i = 1; i <= 300; i++) {
    const isMale = random() > 0.5;
    const gender = isMale ? "Nam" : "Nữ";
    const lastName = lastNames[Math.floor(random() * lastNames.length)];
    const firstName = isMale
      ? firstNames.male[Math.floor(random() * firstNames.male.length)]
      : firstNames.female[Math.floor(random() * firstNames.female.length)];
    const middleName = isMale
      ? middleNames.male[Math.floor(random() * middleNames.male.length)]
      : middleNames.female[Math.floor(random() * middleNames.female.length)];

    const diagnosisData = diagnoses[Math.floor(random() * diagnoses.length)];
    const age = Math.floor(random() * 60) + 20; // Age between 20-79
    const status = statuses[Math.floor(random() * statuses.length)];
    const bloodType = bloodTypes[Math.floor(random() * bloodTypes.length)];

    patients.push({
      id: i,
      name: `${lastName} ${firstName} ${middleName}`,
      age: age,
      gender: gender,
      diagnosis: diagnosisData.name,
      status: status,
      image: diagnosisData.image,
      blood_type: bloodType,
    });
  }

  return patients;
};

// Generate the patients data
const patientsData = generatePatients();

// Create the file content
const fileContent = `export const patientsData = ${JSON.stringify(patientsData, null, 2)};
`;

// Write to file
const outputPath = path.join(__dirname, "src", "constants", "patients.js");
fs.writeFileSync(outputPath, fileContent, "utf8");

console.log(`✅ Successfully generated ${patientsData.length} patients!`);
console.log(`📁 File saved to: ${outputPath}`);
console.log(`\nSample data (first 3 patients):`);
console.log(JSON.stringify(patientsData.slice(0, 3), null, 2));
