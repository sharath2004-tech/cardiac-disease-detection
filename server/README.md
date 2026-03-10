# HeartSense AI Backend

Backend API server for HeartSense AI heart health monitoring application.

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure MongoDB Atlas:**
   - Create a free cluster at [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
   - Get your connection string
   - Update the `MONGODB_URI` in `.env` file

3. **Configure Environment Variables:**
   - Copy `.env.example` to `.env`
   - Update the following variables:
     - `MONGODB_URI`: Your MongoDB Atlas connection string
     - `JWT_SECRET`: A secure random string for JWT tokens
     - `PORT`: Server port (default: 5000)

4. **Start the server:**
   ```bash
   # Development mode with auto-reload
   npm run dev
   
   # Production mode
   npm start
   ```

## MongoDB Atlas Setup

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Sign up or log in
3. Create a new cluster (free tier available)
4. Click "Connect" on your cluster
5. Choose "Connect your application"
6. Copy the connection string
7. Replace `<password>` with your database user password
8. Replace the database name with `heartsense-ai`

Example connection string:
```
mongodb+srv://myuser:mypassword@cluster0.xxxxx.mongodb.net/heartsense-ai?retryWrites=true&w=majority
```

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register a new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user (requires token)

### Patients
- `GET /api/patients` - Get all patients
- `GET /api/patients/:id` - Get patient by ID
- `POST /api/patients` - Create new patient profile
- `PUT /api/patients/:id` - Update patient profile
- `DELETE /api/patients/:id` - Delete patient

### Assessments
- `GET /api/assessments` - Get all assessments
- `GET /api/assessments/patient/:patientId` - Get assessments by patient
- `GET /api/assessments/:id` - Get assessment by ID
- `POST /api/assessments` - Create new assessment
- `PUT /api/assessments/:id` - Update assessment
- `DELETE /api/assessments/:id` - Delete assessment

### Health Check
- `GET /api/health` - Server health check

## Authentication

Protected routes require a JWT token in the Authorization header:
```
Authorization: Bearer <token>
```

## Tech Stack

- **Node.js** - Runtime environment
- **Express.js** - Web framework
- **MongoDB Atlas** - Cloud database
- **Mongoose** - MongoDB ODM
- **JWT** - Authentication
- **bcryptjs** - Password hashing
